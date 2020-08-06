/*
 * vino_yolo.hpp
 *
 *  Created on: July 15th, 2020
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
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <actionlib/server/simple_action_server.h>
#include <robot_vision_msgs/CheckForObjectsAction.h>

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

// Openpose_VINO
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

using namespace InferenceEngine;

namespace object_detection_yolo {
	// 计算物体属于哪一种类别
	// TODO 注意编译过程中static类型变量在跨文件调用中可能存在的问题
	static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
		int n = location / (side * side);
		int loc = location % (side * side);
		return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
	}

	// 创建用于表示bounding_box的结构体
	struct DetectionObject {
		// 结构体成员包括
		// 成员变量：
		// bounding_box的四个角的像素坐标，物体类别，置信度
		int xmin, ymin, xmax, ymax, class_id;
		float confidence;

		// 显式的构造函数，便于生成bounding_box对象
		DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
			this->xmin = static_cast<int>((x - w / 2) * w_scale);
			this->ymin = static_cast<int>((y - h / 2) * h_scale);
			this->xmax = static_cast<int>(this->xmin + w * w_scale);
			this->ymax = static_cast<int>(this->ymin + h * h_scale);
			this->class_id = class_id;
			this->confidence = confidence;
		}
		// 重载运算符, 实现bounding_boxes之间置信度的比较
		bool operator <(const DetectionObject &s2) const {
			return this->confidence < s2.confidence;
		}
		bool operator >(const DetectionObject &s2) const {
			return this->confidence > s2.confidence;
		}
	};

	class YoloParams {
		// 用来加载Yolo参数的类
		// 提供了三种构造函数，便于使用不同的方式加载Yolo参数
		template <typename T>
		void computeAnchors(const std::vector<T>& mask) {
			std::vector<float> maskedAnchors(num * 2);
			for (int i = 0; i < num; ++i) {
				maskedAnchors[i*2] = anchors[mask[i]*2];
				maskedAnchors[i*2+1] = anchors[mask[i]*2+1];
			}
			anchors = maskedAnchors;
		}

	public:
		int num = 0, classes = 0, coords = 0;
		std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};
		// 默认构造函数
		YoloParams() {}
		
		// 自定义构造函数1. 传入RegionYolo对象
		YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
			coords = regionYolo->get_num_coords();
			classes = regionYolo->get_num_classes();
			anchors = regionYolo->get_anchors();
			auto mask = regionYolo->get_mask();
			num = mask.size();

			computeAnchors(mask);
		}
		
		//自定义构造函数2. 传入CNNLayer对象
		YoloParams(CNNLayer::Ptr layer) {
			if (layer->type != "RegionYolo") {
				throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");
			}
			num = layer->GetParamAsInt("num");
			coords = layer->GetParamAsInt("coords");
			classes = layer->GetParamAsInt("classes");

			try {
				anchors = layer->GetParamAsFloats("anchors");
			} catch(const std::exception& e) {
				std::cerr << e.what() << '\n';
			}

			try {
				auto mask = layer->GetParamAsInts("mask");
				num = mask.size();

				computeAnchors(mask);
			} catch (...) {}
		}
	};

	class YoloDetector {
	public:
		YoloParams yoloParams_;
		std::string targetDeviceName_;
		InferenceEngine::Core inferenceEngine_;
		InferenceEngine::CNNNetwork cnnNetwork;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest::Ptr inferRequestCurr;
		// model weights & labels
		bool autoResize_;
		double iouThreshold_;
		double bboxThreshold_;
		char* weights_;
		int image_width = 640;
		int image_height = 480;
		std::vector<std::string> labels;
		// 类构造函数
		YoloDetector();
		// 类析构函数
		~YoloDetector();

		void frameToBlob(const cv::Mat &frame);

		// 加载网络权重和标签文件
		void loadYoloWeights(std::string modelPath_, std::string labelNames_);

		// 配置网络的输入输出
		void setUpNetwork(std::string modelPath_, std::string labelNames_, double iouThreshold_, double bboxThreshold, bool autoResize_);

		void startCurr();

		bool readyCurr();

		// 计算两个识别框的重叠面积(iou)
		double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);

		std::vector<DetectionObject> postProcessCurr(std::vector<DetectionObject> &objects);

		void renderBoundingBoxes(cv::Mat& frame, std::vector<DetectionObject> objects);

		// 解析YoloV3输出的函数
		void parseYoloV3Output(const CNNNetwork &cnnNetwork, const std::string &output_name, 
													 const Blob::Ptr &blob, 
													 const unsigned long resized_im_h,
													 const unsigned long resized_im_w, 
													 const unsigned long original_im_h,
													 const unsigned long original_im_w,
													 const double threshold,
													 std::vector<DetectionObject> &object);

	};
}