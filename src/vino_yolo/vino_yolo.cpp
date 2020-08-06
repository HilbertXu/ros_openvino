/*
 * vino_yolo.cpp
 * 
 *  Created on: July 15th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */
#include <robot_vision_openvino/vino_yolo/vino_yolo.hpp>

using namespace InferenceEngine;

namespace object_detection_yolo {

  YoloDetector::YoloDetector()
    : targetDeviceName_("CPU"),
      autoResize_(false) {
    ROS_INFO("[YoloDetector] Initializing...");
  }

  // 将cv::Mat 格式数据类型转换为长二进制数据类型Blob
	void YoloDetector::frameToBlob(const cv::Mat &frame) {
		// @TODO 设置FLAG_auto_resize
		if (autoResize_) {
			/* Just set input blob containing read image. Resize and layout conversion will be done automatically */
			inferRequestCurr->SetBlob(cnnNetwork.getInputsInfo().begin()->first, wrapMat2Blob(frame));
		} else {
			/* Resize and copy data from the image to the input blob */
			Blob::Ptr frameBlob = inferRequestCurr->GetBlob(cnnNetwork.getInputsInfo().begin()->first);
			matU8ToBlob<uint8_t>(frame, frameBlob);
		}
	}

  void YoloDetector::loadYoloWeights(std::string modelPath_, std::string modelName_, std::string labelNames_) {
    if (modelPath_.empty() || modelName_.empty()) {
      throw std::runtime_error("Invalid model path!!!, .xml, .bin files excepted...");
    }

    modelPath_ += modelName_;
    YoloDetector::weights_ = new char[modelPath_.length() + 1];
    strcpy(YoloDetector::weights_, modelPath_.c_str());

    

    ROS_INFO("[YoloDetector] Loading inference engine...");
    ROS_INFO("[YoloDetector] Device Info: %s", InferenceEngine::GetInferenceEngineVersion()->buildNumber);

    //! Loading network model
    cnnNetwork = YoloDetector::inferenceEngine_.ReadNetwork(weights_);
    //! Loading labels
    std::string labelFileName = fileNameNoExt(weights_) + ".names";

    // Load labels
    ROS_INFO("[YoloDetector] Loading label map...");
    std::ifstream f(labelNames_);
    std::string line;
    if (!f) {
      throw std::runtime_error("Invalid label path: "+ labelNames_);
    }
    while (std::getline(f, line)) {
      labels.push_back(line);
    }
    ROS_INFO("[YoloDetector] %d Categories in total...", labels.size());
  }

  void YoloDetector::setUpNetwork(std::string modelPath_, std::string modelName_, std::string labelNames_) {
    // Load yolo weights first
    loadYoloWeights(modelPath_, modelName_, labelNames_);

    /** YOLOV3-based network should have one input and three output **/
    // --------------------------------- Preparing input blobs ---------------------------------------------
    /*
    using InferenceEngine::InputsDataMap = std::map< std::string, InputInfo::Ptr >
 	  A collection that contains string as key, and InputInfo smart pointer as value. 
    */
    ROS_INFO("[YoloDetector] Checking model inputs...");
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
      throw std::logic_error("This demo accepts networks that have only one input");
    }
    InputInfo::Ptr& input = inputInfo.begin()->second;
    auto inputName = inputInfo.begin()->first;
    input->setPrecision(Precision::U8);
    if (autoResize_) {
      input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
      input->getInputData()->setLayout(Layout::NHWC);
    } else {
      input->getInputData()->setLayout(Layout::NCHW);
    }
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    inSizeVector[0] = 1;  // set batch to 1
    cnnNetwork.reshape(inputShapes);

    // --------------------------------- Preparing output blobs -------------------------------------------
    ROS_INFO("[YoloDetector] Checking model outputs...");
    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto &output : outputInfo) {
      output.second->setPrecision(Precision::FP32);
      output.second->setLayout(Layout::NCHW);
    }

    // Loading model to device
    ROS_INFO("[YoloDetector] Loading model to CPU");
    executableNetwork = inferenceEngine_.LoadNetwork(cnnNetwork, targetDeviceName_);

    // Create infer request
    ROS_INFO("[YoloDetector] Creating infer requests...");
    inferRequestCurr = executableNetwork.CreateInferRequestPtr();
  }

  void YoloDetector::startCurr() {
    inferRequestCurr->StartAsync();
  }

  bool YoloDetector::readyCurr() {
    if (InferenceEngine::OK == inferRequestCurr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
      return true;
    } else {
      return false;
    }
  }

  void YoloDetector::parseYoloV3Output(const CNNNetwork &cnnNetwork, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
      throw std::runtime_error("Invalid size of output " + output_name +
      " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
      ", current W = " + std::to_string(out_blob_h));

    // --------------------------- Extracting layer parameters -------------------------------------
    if (auto ngraphFunction = cnnNetwork.getFunction()) {
      for (const auto op : ngraphFunction->get_ops()) {
        if (op->get_friendly_name() == output_name) {
          auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
          if (!regionYolo) {
            throw std::runtime_error("Invalid output type: " +
                std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
          }

          yoloParams_ = regionYolo;
          break;
        }
      }
    } else {
      throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
      int row = i / side;
      int col = i % side;
      for (int n = 0; n < yoloParams_.num; ++n) {
        int obj_index = EntryIndex(side, yoloParams_.coords, yoloParams_.classes, n * side * side + i, yoloParams_.coords);
        int box_index = EntryIndex(side, yoloParams_.coords, yoloParams_.classes, n * side * side + i, 0);
        float scale = output_blob[obj_index];
        if (scale < threshold)
          continue;
        double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
        double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
        double height = std::exp(output_blob[box_index + 3 * side_square]) * yoloParams_.anchors[2 * n + 1];
        double width = std::exp(output_blob[box_index + 2 * side_square]) * yoloParams_.anchors[2 * n];
        for (int j = 0; j < yoloParams_.classes; ++j) {
          int class_index = EntryIndex(side, yoloParams_.coords, yoloParams_.classes, n * side_square + i, yoloParams_.coords + 1 + j);
          float prob = scale * output_blob[class_index];
          if (prob < threshold)
            continue;
          DetectionObject obj(x, y, height, width, j, prob,
                  static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                  static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
          objects.push_back(obj);
        }
      }
    }
  }

  std::vector<DetectionObject> YoloDetector::postProcessCurr(std::vector<DetectionObject> &objects) {
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
    unsigned long resized_im_h = getTensorHeight(inputDesc);
    unsigned long resized_im_w = getTensorWidth(inputDesc);
    
    // parsing outputs
    for (auto &output : cnnNetwork.getOutputsInfo()) {
      auto output_name = output.first;
      Blob::Ptr blob = inferRequestCurr->GetBlob(output_name);
      // TODO此处改成从rosparam中读取阈值
      parseYoloV3Output(cnnNetwork, output_name, blob, resized_im_h, resized_im_w, image_height, image_width, 0.5, objects);
    } 
    // Filtering overlapping boxes
    std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
    for (size_t i = 0; i < objects.size(); ++i) {
      if (objects[i].confidence == 0)
        continue;
      for (size_t j = i + 1; j < objects.size(); ++j)
        // TODO 此处改成从rosparam中获取iou阈值
        if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
          objects[j].confidence = 0;
    }

    return objects;
  }

  // 计算两个bounding_box的重叠区域占总区域的比例(简单的数学计算...)
	double YoloDetector::IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    // 返回重叠区域占比
    return area_of_overlap / area_of_union;
	}

  void YoloDetector::renderBoundingBoxes(cv::Mat &frame, std::vector<DetectionObject> objects) {
    // Drawing boxes
    for (auto &object : objects) {
      // TODO 此处改成从rosparam中获取识别置信度阈值
      if (object.confidence < 0.3)
        continue;

      auto label = object.class_id;
      float confidence = object.confidence;
      
      if (confidence > 0.3) {
        /** Drawing only objects when >confidence_threshold probability **/
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(3) << confidence;
        cv::putText(frame,
                (label < static_cast<int>(labels.size()) ?
                        labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                    cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                    cv::Scalar(0, 0, 255));
        cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                      cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
      }
    }
  }

  YoloDetector::~YoloDetector() {
    ROS_INFO("[YoloDetector] Stop detecting...");
  }
}
