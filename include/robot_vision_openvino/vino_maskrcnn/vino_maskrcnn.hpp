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
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// robot_control_msgs
#include <robot_control_msgs/Mission.h>
#include <robot_control_msgs/Results.h>
#include <robot_control_msgs/Feedback.h>
#include <robot_vision_msgs/BoundingBoxes.h>
#include <actionlib/server/simple_action_server.h>
#include <robot_vision_msgs/CheckForObjectsAction.h>

// OpenVINO engine
#include <ngraph/ngraph.hpp>
#include <inference_engine.hpp>

// sample plugins
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

using namespace InferenceEngine;

namespace semantic_segmentation_maskrcnn {

  struct MaskBox {
		// 结构体成员包括
		// 成员变量：
		// bounding_box的四个角的像素坐标，物体类别，置信度
		int xmin, ymin, xmax, ymax, box_height, box_width, class_id;
		float confidence;
  };

  // MaskRCNN class
  class MaskRCNN {
  public:
    size_t netBatchSize;
    size_t netInputHeight;
    size_t netInputWidth;
    std::string targetDeviceName_;
    std::string detectionOutputName_;
    std::string maskName_;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::Core inferenceEngine_;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest::Ptr inferRequestCurr;
    // 默认构造函数
    MaskRCNN();
    MaskRCNN(std::string targetDeviceName, std::string detectionOutputName, std::string maskName);
    // 默认析构函数
    ~MaskRCNN();

    // 加载网络权重文件与网络结构文件，并配置网络的输入输出
    void setUpNetwork(std::string modelPath_);

    // 将图像帧转换为模型输入的blob数据
    void frameToBlob(const cv::Mat &frame);

    // 开始当前图像帧的推理
    void startCurr();

    // 判断当前图像帧的推理是否结束
    bool readyCurr();

    // 处理视频帧的识别结果
    cv::Mat postProcessCurr(const cv::Mat &frame, std::vector<MaskBox> &objects);

  };
}