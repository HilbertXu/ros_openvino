/*
 * ros_interface.hpp
 *
 *  Created on: July 31th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */
#pragma once

// C++
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <pthread.h>
#include <functional>
#include <fstream>
#include <chrono>
#include <iterator>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <random>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <boost/thread/shared_mutex.hpp>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// robot_control_msgs
#include <robot_control_msgs/Mission.h>
#include <robot_control_msgs/Results.h>
#include <robot_control_msgs/Feedback.h>

// Interactive face recognition headers
#include <robot_vision_openvino/vino_interactive_face/detector.hpp>
#include <robot_vision_openvino/vino_interactive_face/face.hpp>
#include <robot_vision_openvino/vino_interactive_face/visualizer.hpp>

typedef struct {
  cv::Mat image;
  std_msgs::Header header;
} MatImageWithHeader_;

typedef struct {
int pixel_x, pixel_y;
} PixelCoord_;

class ROSInterface {
public:
  /*!
  * Constructor
  */
  explicit ROSInterface(ros::NodeHandle nh);

  /*!
  * Destructor
  */
  ~ROSInterface();

private:
  // ROS node handle
  ros::NodeHandle nodeHandle_;

  //! Advertise and subscribe to image topics
  image_transport::ImageTransport imageTransport_;

  //！ ROS Publishers & Subscribers
  image_transport::Subscriber imageSubscriber_;
  ros::Publisher controlPublisher_;
  ros::Publisher resultPublisher_;
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
  std::vector<PixelCoord_> pixelCoords_;
  int sumFrame_ = 6;
  // 人脸检测结果
  std::list<Face::Ptr> faces;
  Visualizer::Ptr visualizer;

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

  //! Openpose running on thread
  std::thread mainThread_;

  //! control related flags
  bool FLAG_start_infer;
  bool FLAG_pub_message;
  bool FLAG_no_smooth;
  // control node
  bool underControl_;

  // 是否显示检测图片结果
  bool viewImage_;
  // 是否在控制台中输出检测结果
  bool enableConsoleOutput_;
  // opencv的waitkey Delay
  int waitKeyDelay_;
  // 判断在检测人脸的基础上，是否还对人脸进行进一步处理
  bool isFaceAnalyticsEnabled;

  // 初始化推理引擎
  InferenceEngine::Core inferenceEngine_;

  // parameters for detectors
  std::string targetDevice_;

  // Face detection model
  FaceDetection faceDetector;
  bool enableFaceDetection_;
  std::string faceModelPath_;
  int faceModelBatchSize_;
  bool faceModelRawOutput_;
  bool faceModelAsync_;
  double bb_enlarge_coef;
  double dx_coef;
  double dy_coef;

  // Age-gender model
  AgeGenderDetection ageGenderDetector;
  bool enableAgeGender_;
  std::string ageModelPath_;
  int ageModelBatchSize_;
  bool ageModelRawOutput_;
  bool ageModelAsync_;
  
  // Head pose estimation model
  HeadPoseDetection headPoseDetector;
  bool enableHeadPose_;
  std::string headPoseModelPath_;
  int headPoseModelBatchSize_;
  bool headPoseModelRawOutput_;
  bool headPoseModelAsync_;

  // Emotions recognition
  EmotionsDetection emotionsDetector;
  bool enableEmotions_;
  std::string emotionsModelPath_;
  int emotionsModelBatchSize_;
  bool emotionsModelRawOutput_;
  bool emotionsModelAsync_;
  
  // Facial landmarks
  FacialLandmarksDetection facialLandmarksDetector;
  bool enableFacialLandmarks_;
  std::string facialMarkModelPath_;
  int facialMarkModelBatchSize_;
  bool facialMarkModelRawOutput_;
  bool facialMarkModelAsync_;

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
  void mainFunc();
  
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