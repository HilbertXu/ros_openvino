/* 
 *  Image semantic segmentation, vino_maskrcnn.hpp
 *    created on: Aug 13th, 2020
 *        Author: Hilbert Xu
 *     Institute: Mustar Robot
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
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// robot_control_msgs
#include <robot_control_msgs/Mission.h>
#include <robot_control_msgs/Results.h>
#include <robot_control_msgs/Feedback.h>

// OpenVINO engine
#include <ngraph/ngraph.hpp>
#include <inference_engine.hpp>

// sample plugins
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

using namespace InferenceEngine;

namespace semantic_segment_maskrcnn {
  class MaskRCNN {
    std::string targetDeviceName_;
    std::string detectionOutputName_;
    std::string maskName_;
    InferenceEngine::Core inferenceEngine_;
    InferenceEngine::ExecutableNetwork executableNetwork;
    // MaskRCNN class
  public:
    // 默认构造函数
    MaskRCNN(std::string targetDeviceName, std::string detectionOutputName, std::string maskName);
    // 默认析构函数
    ~MaskRCNN();

    // 加载网络权重文件与网络结构文件，并配置网络的输入输出
    void setUpNetwork(std::string modelPath_);
  }
}