// SPDX-License-Identifier: BSD-2-Clause
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/msg/scan_matching_status.hpp>

namespace hdl_graph_slam {

class ScanMatchingOdometryNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNodelet(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("scan_matching_odometry_nodelet", options) {}

  void onInit() {
    RCLCPP_DEBUG(this->get_logger(), "initializing scan_matching_odometry_nodelet...");

    initialize_params();

    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    if(enable_imu_frontend) {
      std::function<void(geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr)> cbf = std::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, std::placeholders::_1, false);
      msf_pose_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>("msf_core/pose", 1, cbf);
      std::function<void(geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr)> cbt = std::bind(&ScanMatchingOdometryNodelet::msf_pose_callback, this, std::placeholders::_1, true);
      msf_pose_after_update_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>("msf_core/pose_after_update", 1, cbt);
    }

    points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("filtered_points", 256, std::bind(&ScanMatchingOdometryNodelet::cloud_callback, this, std::placeholders::_1));
    read_until_pub = this->create_publisher<std_msgs::msg::Header>("scan_matching_odometry/read_until", 32);
    odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(published_odom_topic, 32);
    trans_pub = this->create_publisher<geometry_msgs::msg::TransformStamped>("scan_matching_odometry/transform", 32);
    status_pub = this->create_publisher<hdl_graph_slam::msg::ScanMatchingStatus>("scan_matching_odometry/status", 8);
    aligned_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_points", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    published_odom_topic = this->declare_parameter<std::string>("published_odom_topic", "/odom");
    points_topic = this->declare_parameter<std::string>("points_topic", "/velodyne_points");
    odom_frame_id = this->declare_parameter<std::string>("odom_frame_id", "odom");
    robot_odom_frame_id = this->declare_parameter<std::string>("robot_odom_frame_id", "robot_odom");

    // The minimum tranlational distance and rotation angle between keyframes.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = this->declare_parameter<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = this->declare_parameter<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = this->declare_parameter<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding
    transform_thresholding = this->declare_parameter<bool>("transform_thresholding", false);
    max_acceptable_trans = this->declare_parameter<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = this->declare_parameter<double>("max_acceptable_angle", 1.0);

    enable_imu_frontend = this->declare_parameter<bool>("enable_imu_frontend", false);
    enable_robot_odometry_init_guess = this->declare_parameter<bool>("enable_robot_odometry_init_guess", false);

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = this->declare_parameter<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }

    registration = select_registration_method(shared_from_this());
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    if(!rclcpp::ok()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);
    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::msg::Header read_until;
    read_until.frame_id = points_topic;
    read_until.stamp = rclcpp::Duration(1, 0) + cloud_msg->header.stamp;
    read_until_pub->publish(read_until);

    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);
  }

  void msf_pose_callback(geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg, bool after_update) {
    if(after_update) {
      msf_pose_after_update = pose_msg;
    } else {
      msf_pose = pose_msg;
    }
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud
   */
  Eigen::Matrix4f matching(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe) {
      prev_time = rclcpp::Time(0);
      prev_trans.setIdentity();
      keyframe_pose.setIdentity();
      keyframe_stamp = stamp;
      keyframe = downsample(cloud);
      registration->setInputTarget(keyframe);
      return Eigen::Matrix4f::Identity();
    }

    auto filtered = downsample(cloud);
    registration->setInputSource(filtered);

    std::string msf_source;
    Eigen::Isometry3f msf_delta = Eigen::Isometry3f::Identity();

    if(enable_imu_frontend) {
      if(msf_pose && rclcpp::Time(msf_pose->header.stamp) > keyframe_stamp &&
         msf_pose_after_update && rclcpp::Time(msf_pose_after_update->header.stamp) > keyframe_stamp) {
        Eigen::Isometry3d pose0, pose1;
        tf2::fromMsg(msf_pose_after_update->pose.pose, pose0);
        tf2::fromMsg(msf_pose->pose.pose, pose1);
        Eigen::Isometry3d delta = pose0.inverse() * pose1;

        msf_source = "imu";
        msf_delta = delta.cast<float>();
      } else {
        std::cerr << "msf data is too old" << std::endl;
      }
    } else if(enable_robot_odometry_init_guess && prev_time.nanoseconds() != 0) {
      geometry_msgs::msg::TransformStamped transform;
      if (tf_buffer->canTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id)) {
        try {
          transform = tf_buffer->lookupTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, robot_odom_frame_id);
        } catch (const tf2::TransformException& e) {
          RCLCPP_INFO(this->get_logger(), "Could not transform of frame %s from time %f to %f", cloud->header.frame_id.c_str(), prev_time.seconds(), stamp.seconds());
        }
      } else if (tf_buffer->canTransform(cloud->header.frame_id, rclcpp::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id)) {
        try {
          transform = tf_buffer->lookupTransform(cloud->header.frame_id, rclcpp::Time(0), cloud->header.frame_id, prev_time, robot_odom_frame_id);
        } catch (const tf2::TransformException& e) {
          RCLCPP_INFO(this->get_logger(), "Could not transform of frame %s from time %f to latest", cloud->header.frame_id.c_str(), prev_time.seconds());
        }
      }

      if(transform.header.stamp.sec == 0 && transform.header.stamp.nanosec == 0) {
        RCLCPP_INFO(this->get_logger(), "failed to look up transform between %s and %s", cloud->header.frame_id.c_str(), robot_odom_frame_id.c_str());
      } else {
        msf_source = "odometry";
        msf_delta = tf2::transformToEigen(transform.transform).cast<float>();
      }
    }

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->align(*aligned, prev_trans * msf_delta.matrix());

    publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

    if(!registration->hasConverged()) {
      RCLCPP_INFO(this->get_logger(), "scan matching has not converged!!");
      RCLCPP_INFO(this->get_logger(), "ignore this frame (time: %f)", stamp.seconds());
      return keyframe_pose * prev_trans;
    }

    Eigen::Matrix4f trans = registration->getFinalTransformation();
    Eigen::Matrix4f odom = keyframe_pose * trans;

    if(transform_thresholding) {
      Eigen::Matrix4f delta = prev_trans.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

      if(dx > max_acceptable_trans || da > max_acceptable_angle) {
        RCLCPP_INFO(this->get_logger(), "too large transform!! %f [m] %f [rad]", dx, da);
        RCLCPP_INFO(this->get_logger(), "ignore this frame (time: %f)", stamp.seconds());
        return keyframe_pose * prev_trans;
      }
    }

    prev_time = stamp;
    prev_trans = trans;

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    tf_broadcaster->sendTransform(keyframe_trans);

    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp).seconds();
    if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
      keyframe = filtered;
      registration->setInputTarget(keyframe);

      keyframe_pose = odom;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans.setIdentity();
    }

    if (aligned_points_pub->get_subscription_count() > 0)
    {
      pcl::transformPointCloud (*cloud, *aligned, odom);
      aligned->header.frame_id = odom_frame_id;
      sensor_msgs::msg::PointCloud2 pc2;
      pcl::toROSMsg(*aligned, pc2);
      aligned_points_pub->publish(pc2);
    }

    return odom;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const rclcpp::Time& stamp, const std::string& base_frame_id, const Eigen::Matrix4f& pose) {
    // publish transform stamped for IMU integration
    geometry_msgs::msg::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, base_frame_id);
    trans_pub->publish(odom_trans);

    // broadcast the transform over tf
    tf_broadcaster->sendTransform(odom_trans);

    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = base_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    odom_pub->publish(odom);
  }

  /**
   * @brief publish scan matching status
   */
  void publish_scan_matching_status(const rclcpp::Time& stamp, const std::string& frame_id, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned, const std::string& msf_source, const Eigen::Isometry3f& msf_delta) {
    if(!status_pub->get_subscription_count()) {
      return;
    }

    hdl_graph_slam::msg::ScanMatchingStatus status;
    status.header.frame_id = frame_id;
    status.header.stamp = stamp;
    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(size_t i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    status.relative_pose = tf2::toMsg(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>()));

    if(!msf_source.empty()) {
      status.prediction_labels.resize(1);
      status.prediction_labels[0].data = msf_source;

      status.prediction_errors.resize(1);
      Eigen::Isometry3f error = Eigen::Isometry3f(registration->getFinalTransformation()).inverse() * msf_delta;
      status.prediction_errors[0] = tf2::toMsg(error.cast<double>());
    }

    status_pub->publish(status);
  }

private:
  // ROS topics

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>::SharedPtr msf_pose_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>::SharedPtr msf_pose_after_update_sub;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr trans_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_points_pub;
  rclcpp::Publisher<hdl_graph_slam::msg::ScanMatchingStatus>::SharedPtr status_pub;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  bool enable_imu_frontend;
  bool enable_robot_odometry_init_guess;
  std::string published_odom_topic;
  std::string points_topic;
  std::string odom_frame_id;
  std::string robot_odom_frame_id;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;

  // keyframe parameters
  double keyframe_delta_trans;  // minimum distance between keyframes
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;

  // odometry calculation
  geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msf_pose;
  geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msf_pose_after_update;

  rclcpp::Time prev_time;
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  rclcpp::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hdl_graph_slam::ScanMatchingOdometryNodelet>();
  node->onInit();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
