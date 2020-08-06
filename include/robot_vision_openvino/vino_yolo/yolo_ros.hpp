/*
 *  HumanPoseEstimate.hpp
 *   Created on: July 15th, 2020
 *       Author: Hilbert Xu
 *    Institute: Mustar Robot
 */

#pragma once    

// vino_yolo
#include <robot_vision_openvino/vino_yolo/vino_yolo.hpp>


namespace object_detection_yolo {
	typedef struct {
		cv::Mat image;
		std_msgs::Header header;
	} MatImageWithHeader_;
		
	class YoloROS {
	public:
		/*!
		* Constructor
		*/
		explicit YoloROS(ros::NodeHandle nh);

		/*!
		* Destructor
		*/
		~YoloROS();

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
		void checkForObjectsActionGoalCB();

		/*!
		* Check for human pose action preempt callback.
		*/
		void checkForObjectsActionPreemptCB();

		/*!
		* Check if a preempt for the check for human pose action has been requested.
		* @return false if preempt has been requested or inactive.
		*/
		bool isCheckingForObjects() const;

		/*!
			* Publishes the detection image
			* @return true if successful
			*/
		bool publishdetectionImage(const cv::Mat& detectionImage);

		// Using
		using CheckForObjectsActionServer = actionlib::SimpleActionServer<robot_vision_msgs::CheckForObjectsAction>;
		using CheckForObjectsActionServerPtr = std::shared_ptr<CheckForObjectsActionServer>;

		// ROS node handle
		ros::NodeHandle nodeHandle_;
		
		// ROS actionlib server
		CheckForObjectsActionServerPtr checkForObjectsActionServer_;

		//! Advertise and subscribe to image topics
		image_transport::ImageTransport imageTransport_;

		//！ ROS Publishers & Subscribers
		image_transport::Subscriber imageSubscriber_;
		ros::Publisher controlPublisher_;
		ros::Publisher bboxesPublisher_;
		ros::Subscriber controlSubscriber_;

		//! Publisher of human pose image
		ros::Publisher detectionImagePublisher_;

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

		//! Detection result
		std::vector<DetectionObject> objects;
		

		//! Camera related parameters
		int frameWidth_;
		int frameHeight_;

		//! Inference engine
		YoloDetector detector;

		//! detection target;
		bool detectSpecificPose_ = false;
		std::string targetPose_;

		//! Openpose running on thread
		std::thread yoloThread_;

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

		// 主函数
		void yolo();
		
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


