/*
 * ros_interface.cpp
 *
 *  Created on: July 31th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_interactive_face/ros_interface.hpp>

// Check for xServer
#include <X11/Xlib.h>

extern "C" {
	double what_time_is_it_now() {
		struct timeval time;
		if (gettimeofday(&time,NULL)) {
			return 0;
		}
		return (double)time.tv_sec + (double)time.tv_usec * .000001;
	}
}

using namespace InferenceEngine;

ROSInterface::ROSInterface(ros::NodeHandle nh)
  : nodeHandle_(nh), imageTransport_(nodeHandle_) {
    ROS_INFO("[InteractiveFace] Node Started!");

    if (!readParameters()) {
      ros::requestShutdown();
    }

    init();
}

ROSInterface::~ROSInterface() {
  {
		boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
		isNodeRunning_ = false;
	}
	mainThread_.join();
}

bool ROSInterface::readParameters() {
	// Load common parameters.
	nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
	nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
	nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

	// Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[ROSInterface] Xserver is running.");
  } else {
    ROS_INFO("[ROSInterface] Xserver is not running.");
    viewImage_ = false;
  }
  std::string modelFolder_;
  std::string ageModelName_;
  std::string faceModelName_;
  std::string headPoseModelName_;
  std::string emotionsModelName_;
  std::string facialMarkModelName_;

  nodeHandle_.param("under_control",               underControl_,             bool(false));

  nodeHandle_.param("base_detector/target_device", targetDevice_,             std::string("CPU"));
  nodeHandle_.param("base_detector/model_folder",  modelFolder_,              std::string("/default"));
  nodeHandle_.param("base_detector/no_smooth",      FLAG_no_smooth,               false);

  nodeHandle_.param("face_detection/enable",          enableFaceDetection_,      true);
  nodeHandle_.param("face_detection/model_name",      faceModelName_,            std::string("/face-detection-adas-0001.xml"));
  nodeHandle_.param("face_detection/batch_size",      faceModelBatchSize_,       16);
  nodeHandle_.param("face_detection/raw_output",      faceModelRawOutput_,       false);
  nodeHandle_.param("face_detection/async",           faceModelAsync_,           false);
  nodeHandle_.param("face_detection/bb_enlarge_coef", bb_enlarge_coef,           double(1.2));
  nodeHandle_.param("face_detection/dx_coef",         dx_coef,                   double(1));
  nodeHandle_.param("face_detection/dy_coef",         dy_coef,                   double(1));

  nodeHandle_.param("age_gender/enable",           enableAgeGender_,          false);
  nodeHandle_.param("age_gender/model_name",       ageModelName_,             std::string("/age-gender-recognition-retail-0013.xml"));
  nodeHandle_.param("age_gender/batch_size",       ageModelBatchSize_,        16);
  nodeHandle_.param("age_gender/raw_output",       ageModelRawOutput_,        false);
  nodeHandle_.param("age_gender/async",            ageModelAsync_,            false);

  nodeHandle_.param("head_pose/enable",            enableHeadPose_,           false);
  nodeHandle_.param("head_pose/model_name",        headPoseModelName_,        std::string("/head-pose-estimation-adas-0001.xml"));
  nodeHandle_.param("head_pose/batch_size",        headPoseModelBatchSize_,   16);
  nodeHandle_.param("head_pose/raw_output",        headPoseModelRawOutput_,   false);
  nodeHandle_.param("head_pose/async",             headPoseModelAsync_,       false);

  nodeHandle_.param("emotions/enable",             enableEmotions_,           false);
  nodeHandle_.param("emotions/model_name",         emotionsModelName_,        std::string("/emotions-recognition-retail-0003.xml"));
  nodeHandle_.param("emotions/batch_size",         emotionsModelBatchSize_,   16);
  nodeHandle_.param("emotions/raw_output",         emotionsModelRawOutput_,   false);
  nodeHandle_.param("emotions/async",              emotionsModelAsync_,       false);

  nodeHandle_.param("facial_landmarks/enable",     enableFacialLandmarks_,    false);
  nodeHandle_.param("facial_landmarks/model_name", facialMarkModelName_,      std::string("/facial-landmarks-35-adas-0002.xml"));
  nodeHandle_.param("facial_landmarks/batch_size", facialMarkModelBatchSize_, 16);
  nodeHandle_.param("facial_landmarks/raw_output", facialMarkModelRawOutput_, false);
  nodeHandle_.param("facial_landmarks/async",      facialMarkModelAsync_,     false);

  faceModelPath_       = modelFolder_ + (faceModelName_);
  ageModelPath_        = modelFolder_ + (ageModelName_);
  headPoseModelPath_   = modelFolder_ + (headPoseModelName_);
  emotionsModelPath_   = modelFolder_ + (emotionsModelName_);
  facialMarkModelPath_ = modelFolder_ + (facialMarkModelName_);

  if (!underControl_) {
    FLAG_start_infer = true;
    FLAG_pub_message = true;
  } else {
    FLAG_start_infer = false;
    FLAG_pub_message = false;
    ROS_INFO("[ROSInterface] Waiting for command from control node...");
  }
	return true;
}


void ROSInterface::init() {
  // Initialize publisher and subscriber
	// sub camera topic properties
	std::string cameraTopicName;
	int cameraQueueSize;
	// pub detection image topic properties
	std::string detectionImageTopicName;
  int detectionImageQueueSize;
	bool detectionImageLatch;
	// sub control topic properties
  std::string subControlTopicName;
  int subControlQueueSize;
	// pub control topic properties
  std::string pubControlTopicName;
  int pubControlQueueSize;
  bool pubControlLatch;
	// pub bounding boxes topic properties
	// std::string pubBboxesTopicName;
  // int pubBboxesQueueSize;
  // bool pubBboxesLatch;

	nodeHandle_.param("subscribers/camera_reading/topic",      cameraTopicName,         std::string("/astra/rgb/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize,         1);
	nodeHandle_.param("publishers/detection_image/topic",      detectionImageTopicName, std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch",      detectionImageLatch,     true);
  nodeHandle_.param("subscribers/control_node/topic",        subControlTopicName,     std::string("/control_to_vision"));
  nodeHandle_.param("subscribers/control_node/queue_size",   subControlQueueSize,     1);
  nodeHandle_.param("publisher/control_node/topic",          pubControlTopicName,     std::string("/vision_to_control"));
  nodeHandle_.param("publisher/control_node/queue_size",     pubControlQueueSize,     1);
  nodeHandle_.param("publisher/control_node/latch",          pubControlLatch,         false);

	// nodeHandle_.param("publisher/bounding_boxes/topic", pubBboxesTopicName, std::string("bounding_boxes"));
  // nodeHandle_.param("publisher/bounding_boxes/queue_size", pubBboxesQueueSize, 1);
  // nodeHandle_.param("publisher/bounding_boxes/latch", pubBboxesLatch, false);
  
	imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &ROSInterface::cameraCallback, this);
	detectionImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
  controlSubscriber_ = nodeHandle_.subscribe(subControlTopicName, subControlQueueSize, &ROSInterface::controlCallback, this);
  controlPublisher_ = 
      nodeHandle_.advertise<robot_control_msgs::Feedback>(pubControlTopicName, pubControlQueueSize, pubControlLatch);
	// bboxesPublisher_ = 
	//     nodeHandle_.advertise<robot_vision_msgs::BoundingBoxes>(pubBboxesTopicName, pubBboxesQueueSize, pubBboxesLatch);

  // 初始化各个检测器
  faceDetector.init(faceModelPath_, targetDevice_, faceModelBatchSize_, 
                             false, faceModelAsync_, 0.5, faceModelRawOutput_, 
                             static_cast<float>(bb_enlarge_coef), static_cast<float>(dx_coef), 
                             static_cast<float>(dy_coef),
                             true);
  ageGenderDetector.init(ageModelPath_, targetDevice_, ageModelBatchSize_,
                                       true, ageModelAsync_, ageModelRawOutput_,
                                       enableAgeGender_);
  headPoseDetector.init(headPoseModelPath_, targetDevice_, headPoseModelBatchSize_,
                                       true, headPoseModelAsync_, headPoseModelRawOutput_,
                                       enableHeadPose_);
  emotionsDetector.init(emotionsModelPath_, targetDevice_, emotionsModelBatchSize_,
                                       true, emotionsModelAsync_, emotionsModelRawOutput_,
                                       enableEmotions_);
  facialLandmarksDetector.init(facialMarkModelPath_, targetDevice_, facialMarkModelBatchSize_,
                                       true, facialMarkModelAsync_, facialMarkModelRawOutput_,
                                       enableFacialLandmarks_);

  ROS_INFO("[ROSInterface] Loading device: %s", inferenceEngine_.GetVersions("CPU"));

  // 将模型加载至推理设备(CPU)
  // 此处true表示所有推理器的处理batch size均为可变值，因为每张图像中的人脸数无法确定
  // 考虑到每次设定batch size可能造成运行速度的下降，后期可以考虑改为设定一个较大的batch size (8 or 16)
  Load(faceDetector).into(inferenceEngine_, targetDevice_, false);
  Load(ageGenderDetector).into(inferenceEngine_, targetDevice_, true);
  Load(headPoseDetector).into(inferenceEngine_, targetDevice_, true);
  Load(emotionsDetector).into(inferenceEngine_, targetDevice_, true);
  Load(facialLandmarksDetector).into(inferenceEngine_, targetDevice_, true);

  visualizer = std::make_shared<Visualizer>(cv::Size(frameWidth_, frameHeight_));

  // start Yolo thread
	mainThread_ = std::thread(&ROSInterface::mainFunc, this);

}

void ROSInterface::controlCallback(const robot_control_msgs::Mission msg) {
	// 接收来自control节点的消息
	if (msg.target == "face") {
		// 如果目标是human pose
		if (msg.action == "detect") {
			// 开始进行姿态估计
			FLAG_start_infer = true;
			// 允许发布检测结果信息
			FLAG_pub_message = true;
			ROS_INFO("[ROSInterface] Start detecting...");
		}
		else if (msg.action == "stop_detect"){
			ROS_INFO("[ROSInterface] Stop inferring...");
			// stop --> 停止对接收到的图像进行推理
			FLAG_start_infer = false;	
			FLAG_pub_message = false;
		}
	}
}

void ROSInterface::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
	ROS_DEBUG("[ROSInterface] USB image received");

	cv_bridge::CvImagePtr cam_image;

	// 将ros格式图像数据转换为cv_bridge::CvImagePtr格式
	try {
		cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	// 如果完成了图像数据的转换，则通过施加独立锁的方式来将图像数据写入临界区
	// 同时更新图像状态
	if (cam_image) {
		{
			// 为mutexImageCallback_加写锁(unique_lock), 并写入图像数据
			boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
			imageHeader_ = msg->header;
			camImageCopy_ = cam_image->image.clone();
		}
		{
			// 为mutexImageStatus加写锁, 并更新当前图像状态，表示已经接收到图像数据
			boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
			imageStatus_ = true;
		}
		// 记录相机参数(图像长宽)
		frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
	}
	return;
}

void ROSInterface::showImageCV(cv::Mat image) {
	cv::imshow("Interactive Face ROS on CPU", image);
}

// 抓取线程，从临界区中抓取数据，并写入缓存区
void* ROSInterface::fetchInThread() {
	{
		// 为mutexImageCallback_对象加读锁，获取读取临界区的权限
		boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
		// 临界区读取函数
		MatImageWithHeader_ imageWithHeader = getMatImageWithHeader();
		// 将临界区的数据写入缓存区
		buff_[buffIndex_] = imageWithHeader.image.clone();
		headerBuff_[buffIndex_] = imageWithHeader.header;
		buffId_[buffIndex_] = actionId_;
	}
}

void* ROSInterface::displayInThread(void* ptr) {
	// 显示缓存区中的图片数据
  ROSInterface::showImageCV(buff_[(buffIndex_ + 1) % 3]);
  int c = cv::waitKey(waitKeyDelay_);
	// 判断waitKey, 是否退出demo
  if (c != -1) c = c % 256;
  if (c == 27) {
    demoDone_ = 1;
    return 0;
  } 
  return 0;
}

void* ROSInterface::estimateInThread() {
	if (FLAG_start_infer) {
    size_t id = 0;
    faceDetector.enqueue(buff_[(buffIndex_+2)%3]);
    faceDetector.submitRequest();
    faceDetector.wait();
    faceDetector.fetchResults();
    auto pre_frame_result = faceDetector.results;

    for (auto &&face : pre_frame_result) {
      if (isFaceAnalyticsEnabled) {
        auto clippedRect = face.location & cv::Rect(0, 0, frameWidth_, frameHeight_);
        cv::Mat face = buff_[(buffIndex_+2)%3](clippedRect);
        ageGenderDetector.enqueue(face);
        headPoseDetector.enqueue(face);
        emotionsDetector.enqueue(face);
        facialLandmarksDetector.enqueue(face);
      }
    }
    if (isFaceAnalyticsEnabled) {
      ageGenderDetector.submitRequest();
      headPoseDetector.submitRequest();
      emotionsDetector.submitRequest();
      facialLandmarksDetector.submitRequest();
    }
    if (isFaceAnalyticsEnabled) {
      ageGenderDetector.wait();
      headPoseDetector.wait();
      emotionsDetector.wait();
      facialLandmarksDetector.wait();
    }
    // Post processing
    std::list<Face::Ptr> prev_faces;
    faces.clear();
    // For every detected face
    for (size_t i = 0; i < pre_frame_result.size(); i++) {
      auto& result = pre_frame_result[i];
      cv::Rect rect = result.location & cv::Rect(0, 0, frameWidth_, frameHeight_);

      Face::Ptr face;
      if (!FLAG_no_smooth) {
          face = matchFace(rect, prev_faces);
          float intensity_mean = calcMean(buff_[(buffIndex_+2)%3](rect));

          if ((face == nullptr) ||
              ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f)) {
              face = std::make_shared<Face>(id++, rect);
          } else {
              prev_faces.remove(face);
          }

          face->_intensity_mean = intensity_mean;
          face->_location = rect;
      } else {
          face = std::make_shared<Face>(id++, rect);
      }

      face->ageGenderEnable((ageGenderDetector.enabled() &&
                              i < ageGenderDetector.maxBatch));
      if (face->isAgeGenderEnabled()) {
          AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
          face->updateGender(ageGenderResult.maleProb);
          face->updateAge(ageGenderResult.age);
      }

      face->emotionsEnable((emotionsDetector.enabled() &&
                            i < emotionsDetector.maxBatch));
      if (face->isEmotionsEnabled()) {
          face->updateEmotions(emotionsDetector[i]);
      }

      face->headPoseEnable((headPoseDetector.enabled() &&
                            i < headPoseDetector.maxBatch));
      if (face->isHeadPoseEnabled()) {
          face->updateHeadPose(headPoseDetector[i]);
      }

      face->landmarksEnable((facialLandmarksDetector.enabled() &&
                              i < facialLandmarksDetector.maxBatch));
      if (face->isLandmarksEnabled()) {
          face->updateLandmarks(facialLandmarksDetector[i]);
      }

      faces.push_back(face);
    }
    visualizer->draw(buff_[(buffIndex_+2)%3], faces);
    //publishInThread();
    if (enableConsoleOutput_) {
			printf("\033[2J");
			printf("\033[1;1H");
			printf("\nFPS:%.1f\n", fps_);
		}
  }
}

void ROSInterface::mainFunc() {
  isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
                                emotionsDetector.enabled() || facialLandmarksDetector.enabled();
  if (emotionsDetector.enabled()) {
    visualizer->enableEmotionBar(emotionsDetector.emotionsVec);
  }

  const auto wait_duration = std::chrono::milliseconds(2000);
	while (!getImageStatus()) {
		printf("Waiting for image.\n");
		if (!isNodeRunning_) {
			return;
		}
		std::this_thread::sleep_for(wait_duration);
	}

	std::thread estimate_thread;
	std::thread fetch_thread;

  srand(22222222);
  {
		// 初始化缓存区内数据, 用将当前帧填满缓存区
		boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
		MatImageWithHeader_ imageWithHeader = getMatImageWithHeader();
		cv::Mat ros_img = imageWithHeader.image.clone();
		buff_[0] = ros_img;
    headerBuff_[0] = imageWithHeader.header;
	}
	buff_[1] = buff_[0].clone();
	buff_[2] = buff_[0].clone();
	headerBuff_[1] = headerBuff_[0];
	headerBuff_[2] = headerBuff_[0];

	int count = 0;

	// 初始化显示图像的窗口
	if (!demoPrefix_ && viewImage_) {
		cv::namedWindow("Interactive Face ROS on CPU", cv::WINDOW_NORMAL);
		cv::moveWindow("Interactive Face ROS on CPU", 0, 0);
		cv::resizeWindow("Interactive Face ROS on CPU", 640, 480);
	}
	
	demoTime_ = what_time_is_it_now();

  while(!demoDone_) {
    // buffIndex_在(0, 1, 2)间循环
		buffIndex_ = (buffIndex_ + 1) % 3;
		// 为fetchInThread函数生成一个线程
		fetch_thread = std::thread(&ROSInterface::fetchInThread, this);
		// 为estimateInThread函数生成一个线程
		estimate_thread = std::thread(&ROSInterface::estimateInThread,this);
		// 计算fps和时间
		fps_ = 1. /(what_time_is_it_now() - demoTime_);
		demoTime_ = what_time_is_it_now();

		// 显示检测图片
		if (viewImage_) {
			displayInThread(0);
		} 

		// 等待fetch_thread 和 estimate_thread完成
		fetch_thread.join();
		estimate_thread.join();
		// 计数
		++count;
		// 如果节点停止运行，终止demo
		if (!isNodeRunning()) {
			demoDone_ = true;
		}
  }
}

MatImageWithHeader_ ROSInterface::getMatImageWithHeader() {
	cv::Mat rosImage = camImageCopy_.clone();
	MatImageWithHeader_ imageWithHeader = {.image=rosImage, .header=imageHeader_};
	return imageWithHeader;
}

bool ROSInterface::getImageStatus(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
	return imageStatus_;
}

bool ROSInterface::isNodeRunning(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
	return isNodeRunning_;
}