/*
 * semantic_segmentation_main.cpp
 * 
 *  Created on: Aug 31th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_maskrcnn/maskrcnn_ros.hpp>

int main(int argc, char** argv) {
	ros::init (argc, argv, "maskrcnn_ros");
	ros::NodeHandle nodeHandle_("~");
	semantic_segmentation_maskrcnn::MaskrcnnROS segmentation(nodeHandle_);

	ros::spin();
	return 0;
}
 
 