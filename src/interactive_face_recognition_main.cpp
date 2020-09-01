/*
 * semantic_segmentation_main.cpp
 * 
 *  Created on: Sep 1st, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_interactive_face/ros_interface.hpp>

int main(int argc, char** argv) {
	ros::init (argc, argv, "interactive_face_ros");
	ros::NodeHandle nodeHandle_("~");
	ROSInterface interface(nodeHandle_);

	ros::spin();
	return 0;
}
 
 