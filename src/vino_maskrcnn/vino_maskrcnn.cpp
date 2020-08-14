/*
 * vino_maskrcnn.cpp
 * 
 *  Created on: Aug 13th, 2020
 *      Author: Hilbert Xu
 *   Institute: Mustar Robot
 */

#include <robot_vision_openvino/vino_maskrcnn/vino_maskrcnn.hpp>

using namespace InferenceEngine;

namespace semantic_segment_maskrcnn {
  MaskRCNN::MaskRCNN(std::string targetDeviceName, std::string detectionOutputName, std::string maskName):
    targetDeviceName_(targetDeviceName),
    detectionOutputName_(detectionOutputName),
    maskName_(maskName) {
      ROS_INFO("[MaskRCNN] Initializing..."
    }
   
}