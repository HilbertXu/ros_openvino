// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <robot_vision_openvino/vino_openpose/human_pose.hpp>

namespace human_pose_estimation {
HumanPose::HumanPose(const std::vector<cv::Point2f>& keypoints,
                     const float& score)
    : keypoints(keypoints),
      score(score) {}
}  // namespace human_pose_estimation
