/* 
 *  Image semantic segmentation, maskrcnn_ros.cpp
 *    created on: Aug 31th, 2020
 *        Author: Hilbert Xu
 *     Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_maskrcnn/maskrcnn_ros.hpp>

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
using namespace semantic_segmentation_maskrcnn;

MaskrcnnROS::MaskrcnnROS(ros::NodeHandle nh) 
: nodeHandle_(nh), imageTransport_(nodeHandle_) {
  ROS_INFO("[MaskrcnnROS] Node started!");
  if (!readParameters()) {
    
    // 如果无法从rosparam服务器中读取到对应的模型参数
    // 则发送关闭节点的命令
    ros::requestShutdown();
  }

  init ();
}

MaskrcnnROS::~MaskrcnnROS() {
  {
  boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
  isNodeRunning_ = false;
  }
  MaskrcnnThread_.join();
}

bool MaskrcnnROS::readParameters() {
  // Load common parameters.
	nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
	nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
	nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

	// Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[MaskrcnnROS] Xserver is running.");
  } else {
    ROS_INFO("[MaskrcnnROS] Xserver is not running.");
    viewImage_ = false;
  }
	return true;
}

void MaskrcnnROS::init() {
  ROS_INFO("[MaskrcnnROS] Initializing...");

	// Initialize weight file for openvino_maskrcnn
	std::string modelName_;
	std::string modelPath_;
	std::string modelPrecision_;
	std::string labelName_;
	std::string labelPath_;
  std::string outputName_;
  std::string maskName_;

	// Load maskrcnn_model parameters from rosparam server
	nodeHandle_.param("maskrcnn_model/name", modelName_, std::string("/default"));
	nodeHandle_.param("maskrcnn_model/folder", modelPath_, std::string("/default"));
	nodeHandle_.param("maskrcnn_model/folder", labelPath_, std::string("/default"));
	nodeHandle_.param("maskrcnn_model/label", labelName_, std::string("/default"));
	nodeHandle_.param("maskrcnn_model/output_name", outputName_, std::string("reshape_do_2d"));
	nodeHandle_.param("maskrcnn_model/mask_name", maskName_, std::string("masks"));

  
  detector.targetDeviceName_    = "CPU";
  detector.detectionOutputName_ = outputName_;
  detector.maskName_            = maskName_;

	modelPath_.append(modelName_);
	labelPath_.append(labelName_);

	nodeHandle_.param("under_control", underControl_, bool(false));

	if (!underControl_) {
		startDetectFlag_ = true;
		pubMessageFlag_ = true;
	} else {
		startDetectFlag_ = false;
		pubMessageFlag_ = false;
		ROS_INFO("[MaskrcnnROS] Waiting for command from control node...");
	}

	// Set up inference engine
	detector.setUpNetwork(modelPath_);
	ROS_INFO("[MaskrcnnROS] Loading label map...");
  std::ifstream f(labelPath_);
  std::string line;
  if (!f) {
    throw std::runtime_error("Invalid label path: "+ labelPath_);
  }
  while (std::getline(f, line)) {
    labels.push_back(line);
  }
  ROS_INFO("[MaskrcnnROS] %d Categories in total...", labels.size());

	// start maskrcnn thread
	MaskrcnnThread_ = std::thread(&MaskrcnnROS::maskrcnn, this);

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
	std::string pubBboxesTopicName;
  int pubBboxesQueueSize;
  bool pubBboxesLatch;

	nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera_top/rgb/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
	nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName, std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);
  nodeHandle_.param("subscribers/control_node/topic", subControlTopicName, std::string("/control_to_vision"));
  nodeHandle_.param("subscribers/control_node/queue_size", subControlQueueSize, 1);
  nodeHandle_.param("publisher/control_node/topic", pubControlTopicName, std::string("/vision_to_control"));
  nodeHandle_.param("publisher/control_node/queue_size", pubControlQueueSize, 1);
  nodeHandle_.param("publisher/control_node/latch", pubControlLatch, false);

	nodeHandle_.param("publisher/bounding_boxes/topic", pubBboxesTopicName, std::string("bounding_boxes"));
  nodeHandle_.param("publisher/bounding_boxes/queue_size", pubBboxesQueueSize, 1);
  nodeHandle_.param("publisher/bounding_boxes/latch", pubBboxesLatch, false);
  
	imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &MaskrcnnROS::cameraCallback, this);
	detectionImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
  controlSubscriber_ = nodeHandle_.subscribe(subControlTopicName, subControlQueueSize, &MaskrcnnROS::controlCallback, this);
  controlPublisher_ = 
      nodeHandle_.advertise<robot_control_msgs::Feedback>(pubControlTopicName, pubControlQueueSize, pubControlLatch);
	bboxesPublisher_ = 
	    nodeHandle_.advertise<robot_vision_msgs::BoundingBoxes>(pubBboxesTopicName, pubBboxesQueueSize, pubBboxesLatch);

	// Action servers
	std::string checkForObjectsActionName;
	nodeHandle_.param("action/camera_reading/topic", checkForObjectsActionName, std::string("check_for_human_pose"));
	checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));

  checkForObjectsActionServer_->registerGoalCallback(boost::bind(&MaskrcnnROS::checkForObjectsActionGoalCB, this));

  checkForObjectsActionServer_->registerPreemptCallback(boost::bind(&MaskrcnnROS::checkForObjectsActionPreemptCB, this));

  checkForObjectsActionServer_->start();
}

void MaskrcnnROS::controlCallback(const robot_control_msgs::Mission msg) {
	// 接收来自control节点的消息
	if (msg.target == "object") {
		// 如果目标是human pose
		if (msg.action == "detect") {
			if (msg.attributes.object.name != "") {
				// 有检测目标 --> 检测特定物体
				detectSpecificObject_ = true;
				// 记录目标物体
				targetObject_ = msg.attributes.object.name;
				ROS_INFO("[MaskrcnnROS] Detect target: %s", targetObject_.c_str());
			}
			// 开始进行物体检测
			startDetectFlag_ = true;
			// 允许发布检测结果信息
			pubMessageFlag_ = true;
			ROS_INFO("[MaskrcnnROS] Start detecting...");
		}
		else if (msg.action == "stop_detect"){
			ROS_INFO("[MaskrcnnROS] Stop inferring...");
			// stop --> 停止对接收到的图像进行推理
			startDetectFlag_ = false;	
			pubMessageFlag_ = false;
		}
	}
}

void MaskrcnnROS::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
	ROS_DEBUG("[MaskrcnnROS] USB image received");

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

void MaskrcnnROS::checkForObjectsActionGoalCB() {
	ROS_DEBUG("[MaskrcnnROS] Start check for human poses action");

	boost::shared_ptr<const robot_vision_msgs::CheckForObjectsGoal> imageActionPtr = checkForObjectsActionServer_->acceptNewGoal();
	sensor_msgs::Image imageAction = imageActionPtr->image;

	cv_bridge::CvImagePtr cam_image;

	try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
	if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void MaskrcnnROS::checkForObjectsActionPreemptCB() {
  ROS_DEBUG("[MaskrcnnROS] Preempt check for human poses action.");
  checkForObjectsActionServer_->setPreempted();
}

bool MaskrcnnROS::isCheckingForObjects() const {
  return (ros::ok() && checkForObjectsActionServer_->isActive() && !checkForObjectsActionServer_->isPreemptRequested());
}

void MaskrcnnROS::showImageCV(cv::Mat image) {
	cv::imshow("Mask RCNN ROS on CPU", image);
}

// 抓取线程，从临界区中抓取数据，并写入缓存区
void* MaskrcnnROS::fetchInThread() {
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

void* MaskrcnnROS::displayInThread(void* ptr) {
	// 显示缓存区中的图片数据
  MaskrcnnROS::showImageCV(buff_[(buffIndex_ + 1) % 3]);
  int c = cv::waitKey(waitKeyDelay_);
	// 判断waitKey, 是否退出demo
  if (c != -1) c = c % 256;
  if (c == 27) {
    demoDone_ = 1;
    return 0;
  } 
  return 0;
}

void* MaskrcnnROS::estimateInThread() {
	objects.clear();
	if (startDetectFlag_) {
		detector.frameToBlob(buff_[(buffIndex_+2)%3]);
		detector.startCurr();
		while (true) {
			if (detector.readyCurr()) {
				break;
			}
		}
		buff_[(buffIndex_+2)%3] = detector.postProcessCurr(buff_[(buffIndex_+2)%3], objects).clone();

		if (enableConsoleOutput_) {
			printf("\033[2J");
			printf("\033[1;1H");
			printf("\nFPS:%.1f\n", fps_);
			printf("Object:\n\n");
			for (auto object: objects) {
				printf ("%s\t %.2f\n", labels[object.class_id].c_str(), object.confidence);
			}
		}
		// 发布识别结果
		//publishInThread();
	}
}

void MaskrcnnROS::maskrcnn() {
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
		cv::namedWindow("Maskrcnn ROS on CPU", cv::WINDOW_NORMAL);
		cv::moveWindow("Maskrcnn ROS on CPU", 0, 0);
		cv::resizeWindow("Maskrcnn ROS on CPU", 640, 480);
	}
	
	demoTime_ = what_time_is_it_now();

	while (!demoDone_) {
		// buffIndex_在(0, 1, 2)间循环
		buffIndex_ = (buffIndex_ + 1) % 3;
		// 为fetchInThread函数生成一个线程
		fetch_thread = std::thread(&MaskrcnnROS::fetchInThread, this);
		// 为estimateInThread函数生成一个线程
		estimate_thread = std::thread(&MaskrcnnROS::estimateInThread, this);
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

MatImageWithHeader_ MaskrcnnROS::getMatImageWithHeader() {
	cv::Mat rosImage = camImageCopy_.clone();
	MatImageWithHeader_ imageWithHeader = {.image=rosImage, .header=imageHeader_};
	return imageWithHeader;
}

bool MaskrcnnROS::getImageStatus(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
	return imageStatus_;
}

bool MaskrcnnROS::isNodeRunning(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
	return isNodeRunning_;
}