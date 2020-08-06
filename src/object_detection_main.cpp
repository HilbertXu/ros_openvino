/*
 * human_pose_estimator.cpp
 * 
 *  Created on: July 15th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <ros/ros.h>
#include <robot_vision_openvino/vino_yolo/yolo_ros.hpp>

int main(int argc, char** argv) {
	ros::init (argc, argv, "yolo_ros");
	ros::NodeHandle nodeHandle_("~");
	object_detection_yolo::YoloROS yoloROS(nodeHandle_);

	ros::spin();
	return 0;
}
 
 