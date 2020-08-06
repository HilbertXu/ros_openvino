/*
 *  HumanPoseEstimate.hpp
 *   Created on: July 15th, 2020
 *       Author: Hilbert Xu
 *    Institute: Mustar Robot
 */

#pragma once    

// c++
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sys/time.h>
#include <boost/thread/shared_mutex.hpp>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <actionlib/server/simple_action_server.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// robot_control_msgs
#include <robot_control_msgs/Mission.h>
#include <robot_control_msgs/Results.h>
#include <robot_control_msgs/Feedback.h>
#include <robot_vision_msgs/CheckForHumanPosesAction.h>

// OpenVINO engine
#include <inference_engine.hpp>

// Openpose_VINO
#include <samples/ocv_common.hpp>
#include <robot_vision_openvino/vino_openpose/render_human_pose.hpp>
#include <robot_vision_openvino/vino_openpose/human_pose_estimator.hpp>

namespace human_pose_estimation {
	typedef struct {
		cv::Mat image;
		std_msgs::Header header;
	} MatImageWithHeader_;
		
	class OpenposeROS {
	public:
		/*!
		* Constructor
		*/
		explicit OpenposeROS(ros::NodeHandle nh);

		/*!
		* Destructor
		*/
		~OpenposeROS();

	private:
		/*!
			* Reads and verifies ROS parameters
			* @return true if successful
			*/
		bool readParameters();

		/*!
			* Initialize ROS connections
			*/
		void init();

		/*!
			* Callback of camera
			* @param[in] msg image pointer
			*/
		void cameraCallback(const sensor_msgs::ImageConstPtr& msg);

		/*!
			* Callback of control node
			* @param[in] msg control Mission
			*/
		void controlCallback(const robot_control_msgs::Mission msg);

		/*!
		* Check for human pose action goal callback.
		*/
		void checkForHumanPosesActionGoalCB();

		/*!
		* Check for human pose action preempt callback.
		*/
		void checkForHumanPosesActionPreemptCB();

		/*!
		* Check if a preempt for the check for human pose action has been requested.
		* @return false if preempt has been requested or inactive.
		*/
		bool isCheckingForHumanPoses() const;

		/*!
			* Publishes the Estimation image
			* @return true if successful
			*/
		bool publishEstimationImage(const cv::Mat& EstimationImage);

		// Using
		using CheckForHumanPosesActionServer = actionlib::SimpleActionServer<robot_vision_msgs::CheckForHumanPosesAction>;
		using CheckForHumanPosesActionServerPtr = std::shared_ptr<CheckForHumanPosesActionServer>;

		// ROS node handle
		ros::NodeHandle nodeHandle_;
		
		// ROS actionlib server
		CheckForHumanPosesActionServerPtr checkForHumanPosesActionServer_;

		//! Advertise and subscribe to image topics
		image_transport::ImageTransport imageTransport_;

		//！ ROS Publishers & Subscribers
		image_transport::Subscriber imageSubscriber_;
		ros::Publisher controlPublisher_;
		ros::Subscriber controlSubscriber_;

		//! Publisher of human pose image
		ros::Publisher estimationImagePublisher_;

		// 临界读写区，储存从相机话题中获取到的图片以及检测结果图片
		// 通过加锁的方式来获取进行读写操作的权限
		std_msgs::Header imageHeader_;
		cv::Mat camImageCopy_;

		// 缓存区，
		// 使用fetch线程通过加读锁的方式, 抓取临界区数据(图像，header)写入缓存区
		//! buff_中储存cv::Mat对象
		cv::Mat buff_[3];
		std_msgs::Header headerBuff_[3];
		// 储存符合网络输入尺寸的图像数据
		cv::Mat resizedBuff_[3];
		int buffId_[3];
		int buffIndex_ = 0;

		// demo 相关参数
		char* demoPrefix_;
		float fps_ = 0;
		int demoDelay_ = 0;
		int demoFrame_ = 0;
		int demoDone_ = 0;
		int demoIndex_ = 0;
		int demoTotal_ = 0;
		double demoTime_;
		

		//! Camera related parameters
		int frameWidth_;
		int frameHeight_;

		//! Inference engine
		HumanPoseEstimator estimator;

		//! Estimation target;
		bool detectSpecificPose_ = false;
		std::string targetPose_;

		//! Openpose running on thread
		std::thread openposeThread_;

		//! control related flags
		bool startEstimateFlag_;
		bool pubMessageFlag_;

		// 是否显示检测图片结果
		bool viewImage_;
		// 是否在控制台中输出检测结果
		bool enableConsoleOutput_;
		// opencv的waitkey Delay
		int waitKeyDelay_;

		// 初始化全局shared_mutex对象
		boost::shared_mutex mutexImageCallback_;

		bool imageStatus_ = false;
		boost::shared_mutex mutexImageStatus_;

		bool isNodeRunning_ = true;
		boost::shared_mutex mutexNodeStatus_;

		int actionId_;
		boost::shared_mutex mutexActionStatus_;

		// 显示检测图片函数
		void showImageCV(cv::Mat image);

		// 从相机话题中抓取待识别图片的线程
		void* fetchInThread();

		// 显示检测结果图片的线程
		void* displayInThread(void* ptr);

		// openpose推理线程
		void* estimateInThread();

		// 初始化推理引擎函数
		void setUpInferenceEngine();

		// 主函数
		void openpose();
		
		// 返回带header的图片结构
		MatImageWithHeader_ getMatImageWithHeader();

		// 检查当前图片临界区的状态, 并加上共享锁, 确保其他线程可以施加读锁
		bool getImageStatus(void);

		// 检查当前节点状态, 并加上共享锁
		bool isNodeRunning(void);

		// 发布检测结果的线程
		void* publishInThread();

		// 显示循环
		void* displayLoop(void* ptr);

		// 推理循环
		void* estimateLoop(void* ptr);

	};
}


