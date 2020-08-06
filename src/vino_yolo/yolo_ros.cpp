/*
 * openpose_ros.cpp
 *  Created on: July 15th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_yolo/yolo_ros.hpp>

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

using namespace object_detection_yolo;
using namespace InferenceEngine;


char* weights;

YoloROS::YoloROS(ros::NodeHandle nh)
: nodeHandle_(nh), imageTransport_(nodeHandle_) {
	ROS_INFO("[YoloROS] Node started!");

	if (!readParameters()) {
		// 如果无法从rosparam服务器中读取到对应的模型参数
		// 则发送关闭节点的命令
		ros::requestShutdown();
	}

	init();
}

YoloROS::~YoloROS() {
	{
		boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
		isNodeRunning_ = false;
	}
	yoloThread_.join();
}

bool YoloROS::readParameters() {
	// Load common parameters.
	nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
	nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
	nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

	// Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloROS] Xserver is running.");
  } else {
    ROS_INFO("[YoloROS] Xserver is not running.");
    viewImage_ = false;
  }
	return true;
}

void YoloROS::init() {
	ROS_INFO("[YoloROS] Initializing...");

	// Initialize weight file for openvino_openpose
	std::string modelPath_;
	std::string modelName_;
	std::string labelPath_;

	nodeHandle_.param("yolo_weights_path", modelPath_, std::string("$(find robot_vision_openvino)/models/yolo/Yolov3-tiny/FP32/"));
	nodeHandle_.param("yolo_weights_name", modelName_, std::string("frozen_yolov3_tiny.xml"));
	nodeHandle_.param("yolo_label_names", labelPath_, std::string("/default"));

	// weightsPath += weightsModel;
	// weights = new char[weightsPath.length() + 1];
	// strcpy(weights, weightsPath.c_str());

	// Set up inference engine
	detector.setUpNetwork(modelPath_, modelName_, labelPath_);

	// start openpose thread
	yoloThread_ = std::thread(&YoloROS::yolo, this);

	// Initialize publisher and subscriber
	// sub camera topic properties
	std::string cameraTopicName;
	int cameraQueueSize;
	// pub estimation image topic properties
	std::string estimationImageTopicName;
  int estimationImageQueueSize;
	bool estimationImageLatch;
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

	nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/astra/rgb/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
	nodeHandle_.param("publishers/estimation_image/topic", estimationImageTopicName, std::string("estimation_image"));
  nodeHandle_.param("publishers/estimation_image/queue_size", estimationImageQueueSize, 1);
  nodeHandle_.param("publishers/estimation_image/latch", estimationImageLatch, true);
  nodeHandle_.param("subscribers/control_node/topic", subControlTopicName, std::string("/control_to_vision"));
  nodeHandle_.param("subscribers/control_node/queue_size", subControlQueueSize, 1);
  nodeHandle_.param("publisher/control_node/topic", pubControlTopicName, std::string("/vision_to_control"));
  nodeHandle_.param("publisher/control_node/queue_size", pubControlQueueSize, 1);
  nodeHandle_.param("publisher/control_node/latch", pubControlLatch, false);

	nodeHandle_.param("publisher/bounding_boxes/topic", pubBboxesTopicName, std::string("bounding_boxes"));
  nodeHandle_.param("publisher/bounding_boxes/queue_size", pubBboxesQueueSize, 1);
  nodeHandle_.param("publisher/bounding_boxes/latch", pubBboxesLatch, false);
  
	imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &YoloROS::cameraCallback, this);
	estimationImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(estimationImageTopicName, estimationImageQueueSize, estimationImageLatch);
  controlSubscriber_ = nodeHandle_.subscribe(subControlTopicName, subControlQueueSize, &YoloROS::controlCallback, this);
  controlPublisher_ = 
      nodeHandle_.advertise<robot_control_msgs::Feedback>(pubControlTopicName, pubControlQueueSize, pubControlLatch);
	bboxesPublisher_ = 
	    nodeHandle_.advertise<robot_vision_msgs::BoundingBoxes>(pubBboxesTopicName, pubBboxesQueueSize, pubBboxesLatch);

	// Action servers
	std::string checkForObjectsActionName;
	nodeHandle_.param("action/camera_reading/topic", checkForObjectsActionName, std::string("check_for_human_pose"));
	checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));

  checkForObjectsActionServer_->registerGoalCallback(boost::bind(&YoloROS::checkForObjectsActionGoalCB, this));

  checkForObjectsActionServer_->registerPreemptCallback(boost::bind(&YoloROS::checkForObjectsActionPreemptCB, this));

  checkForObjectsActionServer_->start();
}

void YoloROS::controlCallback(const robot_control_msgs::Mission msg) {
	// 接收来自control节点的消息
	if (msg.target == "human_pose") {
		// 如果目标是human pose
		if (msg.action == "detect" && msg.attributes.human.gesture != "") {
			// 有检测目标 --> 检测特定姿态
			detectSpecificPose_ = true;
			// 记录目标姿态
			targetPose_ = msg.attributes.human.gesture;
			// 开始进行姿态估计
			startEstimateFlag_ = true;
			// 允许发布检测结果信息
			pubMessageFlag_ = true;
		}
		else if (msg.action == "estimate") {
			// 没有检测目标 --> 返回每帧图像中所有人体骨架的关键点以及姿态
			detectSpecificPose_ = false;
			// 开始进行姿态估计
			startEstimateFlag_ = true;
			// 允许发布检测结果信息
			pubMessageFlag_ = true;
		}
		else if (msg.action.find("stop")){
			// stop --> 停止对接收到的图像进行推理
			startEstimateFlag_ = false;	
		}
	}
}

void YoloROS::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
	ROS_DEBUG("[YoloROS] USB image received");

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

void YoloROS::checkForObjectsActionGoalCB() {
	ROS_DEBUG("[YoloROS] Start check for human poses action");

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

void YoloROS::checkForObjectsActionPreemptCB() {
  ROS_DEBUG("[YoloROS] Preempt check for human poses action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloROS::isCheckingForObjects() const {
  return (ros::ok() && checkForObjectsActionServer_->isActive() && !checkForObjectsActionServer_->isPreemptRequested());
}

void YoloROS::showImageCV(cv::Mat image) {
	cv::imshow("YoloV3 ROS on CPU", image);
}

// 抓取线程，从临界区中抓取数据，并写入缓存区
void* YoloROS::fetchInThread() {
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
	//! TODO
	//! 根据OpenVINO的推理引擎要求，对缓存区内的图像进行预处理，(bgr8 -> rgb, resize)
}

void* YoloROS::displayInThread(void* ptr) {
	// 显示缓存区中的图片数据
  YoloROS::showImageCV(buff_[(buffIndex_ + 1) % 3]);
  int c = cv::waitKey(waitKeyDelay_);
	// 判断waitKey, 是否退出demo
  if (c != -1) c = c % 256;
  if (c == 27) {
    demoDone_ = 1;
    return 0;
  } 
  return 0;
}

void* YoloROS::estimateInThread() {
	objects.clear();
	detector.frameToBlob(buff_[(buffIndex_+2)%3]);
	detector.startCurr();
	while (true) {
		if (detector.readyCurr()) {
			break;
		}
	}
	detector.postProcessCurr(objects);
	// 发布识别结果
	publishInThread();
	// 绘制识别框
	detector.renderBoundingBoxes(buff_[(buffIndex_+2)%3], objects);
}

void YoloROS::yolo() {
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
		cv::namedWindow("YoloV3 ROS on CPU", cv::WINDOW_NORMAL);
		cv::moveWindow("YoloV3 ROS on CPU", 0, 0);
		cv::resizeWindow("YoloV3 ROS on CPU", 640, 480);
	}
	
	demoTime_ = what_time_is_it_now();

	while (!demoDone_) {
		// buffIndex_在(0, 1, 2)间循环
		buffIndex_ = (buffIndex_ + 1) % 3;
		// 为fetchInThread函数生成一个线程
		fetch_thread = std::thread(&YoloROS::fetchInThread, this);
		// 为estimateInThread函数生成一个线程
		estimate_thread = std::thread(&YoloROS::estimateInThread, this);
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

MatImageWithHeader_ YoloROS::getMatImageWithHeader() {
	cv::Mat rosImage = camImageCopy_.clone();
	MatImageWithHeader_ imageWithHeader = {.image=rosImage, .header=imageHeader_};
	return imageWithHeader;
}

bool YoloROS::getImageStatus(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
	return imageStatus_;
}

bool YoloROS::isNodeRunning(void) {
	boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
	return isNodeRunning_;
}

void* YoloROS::publishInThread() {
	robot_vision_msgs::BoundingBoxes msg;
	msg.image_header.frame_id = "/camera_top_rgb_frame";
	for (auto object: objects) {
		if (object.confidence < 0.5) {
			continue;
		} else {
			robot_vision_msgs::BoundingBox bbox_;
			bbox_.Class = detector.labels[object.class_id];
			bbox_.probability = object.confidence;
			bbox_.xmin = object.xmin;
			bbox_.xmax = object.xmax;
			bbox_.ymin = object.ymin;
			bbox_.ymax = object.ymax;
			msg.bounding_boxes.push_back(bbox_);
		}
	}
	bboxesPublisher_.publish(msg);
}

