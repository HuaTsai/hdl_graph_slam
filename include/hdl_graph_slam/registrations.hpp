// SPDX-License-Identifier: BSD-2-Clause

#ifndef HDL_GRAPH_SLAM_REGISTRATIONS_HPP
#define HDL_GRAPH_SLAM_REGISTRATIONS_HPP

#include <rclcpp/rclcpp.hpp>

#include <pcl/registration/registration.h>

namespace hdl_graph_slam {

/**
 * @brief select a scan matching algorithm according to rosparams
 * @param pnh
 * @return selected scan matching
 */
pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr select_registration_method(rclcpp::Node *node);

}  // namespace hdl_graph_slam

#endif  //
