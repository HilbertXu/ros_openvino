/*
 * human_pose_estimate_main.cpp
 * 
 *  Created on: July 15th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <ros/ros.h>
#include <robot_vision_openvino/vino_openpose/openpose_ros.hpp>

int main(int argc, char** argv) {
	ros::init (argc, argv, "openpose_ros");
	ros::NodeHandle nodeHandle_("~");
	human_pose_estimation::OpenposeROS openposeROS(nodeHandle_);

	ros::spin();
	return 0;
}
 
 