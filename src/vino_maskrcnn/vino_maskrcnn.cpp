/*
 * vino_maskrcnn.cpp
 * 
 *  Created on: Aug 13th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */
#include <robot_vision_openvino/vino_maskrcnn/vino_maskrcnn.hpp>

using namespace InferenceEngine;

namespace semantic_segmentation_maskrcnn {
  MaskRCNN::MaskRCNN() { ROS_INFO("[MaskRCNN] Initializing..."); }
  MaskRCNN::MaskRCNN(std::string targetDeviceName, std::string detectionOutputName, std::string maskName):
    targetDeviceName_(targetDeviceName),
    detectionOutputName_(detectionOutputName),
    maskName_(maskName) {
      ROS_INFO("[MaskRCNN] Initializing...");
    }
  
  MaskRCNN::~MaskRCNN() {
    ROS_INFO("[MaskRCNN] Shutting down...");
  }

  void MaskRCNN::setUpNetwork(std::string modelPath_) {
    ROS_INFO("[MaskRCNN] Loading inference engine...");
    ROS_INFO("[MaskRCNN] Device info:  %s", InferenceEngine::GetInferenceEngineVersion()->buildNumber);
    ROS_INFO("[MaskRCNN] Loading model from %s", modelPath_.c_str());
    network = inferenceEngine_.ReadNetwork(modelPath_);
    // add DetectionOutput layer as output so we can get detected boxes and their probabilities
    network.addOutput(detectionOutputName_.c_str(), 0);

    // 设置模型输入接口    
    ROS_INFO("[MaskRCNN] Setting up network input blobs...");
    InputsDataMap inputInfo(network.getInputsInfo());
    std::string imageInputName;
    for (const auto & inputInfoItem : inputInfo) {
      if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
        imageInputName = inputInfoItem.first;
        inputInfoItem.second->setPrecision(Precision::U8);
      } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
        inputInfoItem.second->setPrecision(Precision::FP32);
      } else {
        throw std::logic_error("Unsupported input shape with size = " + std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()));
      }
    }
    /** network dimensions for image input **/
    // 设置网络的图像输入格式
    const TensorDesc& inputDesc = inputInfo[imageInputName]->getTensorDesc();
    IE_ASSERT(inputDesc.getDims().size() == 4);
    netBatchSize = getTensorBatch(inputDesc);
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);
    ROS_INFO("[MaskRCNN] Model input info: batch_size %d, input_height %d, input_width %d", netBatchSize, netInputHeight, netInputWidth);

    // 配置模型的输出端
    ROS_INFO("[MaskRCNN] Setting up networks output blobs...");
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    for (auto & item : outputInfo) {
      item.second->setPrecision(Precision::FP32);
    }

    // Load model to the device
    ROS_INFO("[MaskRCNN] Loading model to %s", targetDeviceName_.c_str());
    executableNetwork = inferenceEngine_.LoadNetwork(network, targetDeviceName_);

    // Create infer request
    inferRequestCurr = executableNetwork.CreateInferRequestPtr();
  }

  void MaskRCNN::frameToBlob(const cv::Mat &frame) {
    InputsDataMap inputInfo(network.getInputsInfo());
    for (const auto & inputInfoItem : inputInfo) {
      Blob::Ptr input = inferRequestCurr->GetBlob(inputInfoItem.first);

      if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {
        matU8ToBlob<unsigned char>(frame, input);
      }

      /** Fill second input tensor with image info **/
      if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        data[0] = static_cast<float>(netInputHeight);  // height
        data[1] = static_cast<float>(netInputWidth);  // width
        data[2] = 1;
      }
    }
  }

  void MaskRCNN::startCurr() {
    inferRequestCurr->StartAsync();
  }

  bool MaskRCNN::readyCurr() {
    if (InferenceEngine::OK == inferRequestCurr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
      return true;
    } else {
      return false;
    }
  }

  cv::Mat MaskRCNN::postProcessCurr(const cv::Mat &frame, std::vector<MaskBox> &objects) {
    const auto do_blob = inferRequestCurr->GetBlob(detectionOutputName_.c_str());
    const auto do_data = do_blob->buffer().as<float*>();

    const auto masks_blob = inferRequestCurr->GetBlob(maskName_.c_str());
    const auto masks_data = masks_blob->buffer().as<float*>();

    const float PROBABILITY_THRESHOLD = 0.2f;
    const float MASK_THRESHOLD = 0.5f;

    IE_ASSERT(do_blob->getTensorDesc().getDims().size() == 2);
    size_t BOX_DESCRIPTION_SIZE = do_blob->getTensorDesc().getDims().back();

    const TensorDesc& masksDesc = masks_blob->getTensorDesc();
    IE_ASSERT(masksDesc.getDims().size() == 4);
    size_t BOXES = getTensorBatch(masksDesc);
    size_t C = getTensorChannels(masksDesc);
    size_t H = getTensorHeight(masksDesc);
    size_t W = getTensorWidth(masksDesc);

    size_t box_stride = W * H * C;
    std::map<size_t, size_t> class_color;

    cv::Mat output_image = frame.clone();

    for (size_t box = 0; box < BOXES; box++) {
      float* box_info = do_data + box * BOX_DESCRIPTION_SIZE;
      auto batch = static_cast<int>(box_info[0]);
      if (batch < 0)
        break;
      if (batch >= static_cast<int>(netBatchSize))
        throw std::logic_error("Invalid batch ID within detection output box");
      float prob = box_info[2];
      float x1 = std::min(std::max(0.0f, box_info[3] * frame.cols), static_cast<float>(frame.cols));
      float y1 = std::min(std::max(0.0f, box_info[4] * frame.rows), static_cast<float>(frame.rows));
      float x2 = std::min(std::max(0.0f, box_info[5] * frame.cols), static_cast<float>(frame.cols));
      float y2 = std::min(std::max(0.0f, box_info[6] * frame.rows), static_cast<float>(frame.rows));
      int box_width = std::min(static_cast<int>(std::max(0.0f, x2 - x1)), frame.cols);
      int box_height = std::min(static_cast<int>(std::max(0.0f, y2 - y1)), frame.rows);
      auto class_id = static_cast<size_t>(box_info[1] + 1e-6f);
      if (prob > PROBABILITY_THRESHOLD) {
        // 记录每帧图像中的识别结果
        MaskBox tempBox;
        tempBox.class_id = class_id;
        tempBox.xmin = x1;
        tempBox.ymin = y1;
        tempBox.xmax = x2;
        tempBox.ymax = y2;
        tempBox.confidence = prob;
        tempBox.box_height = box_height;
        tempBox.box_width = box_width;
        objects.push_back(tempBox);

        // 将mask绘制到原始图片上
        size_t color_index = class_color.emplace(class_id, class_color.size()).first->second;
        auto& color = CITYSCAPES_COLORS[color_index % arraySize(CITYSCAPES_COLORS)];
        float* mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
        cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

        cv::Rect roi = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height);
        cv::Mat roi_input_img = output_image(roi);
        const float alpha = 0.7f;

        cv::Mat resized_mask_mat(box_height, box_width, CV_32FC1);
        cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width, box_height));

        cv::Mat uchar_resized_mask(box_height, box_width, CV_8UC3,
            cv::Scalar(color.blue(), color.green(), color.red()));
        roi_input_img.copyTo(uchar_resized_mask, resized_mask_mat <= MASK_THRESHOLD);

        cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
        cv::rectangle(output_image, roi, cv::Scalar(0, 0, 1), 1);
      }
    }
    return output_image;
  }
}