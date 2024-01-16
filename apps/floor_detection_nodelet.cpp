// SPDX-License-Identifier: BSD-2-Clause

#include <rclcpp/rclcpp.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <hdl_graph_slam/msg/floor_coeffs.hpp>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/plane_clipper3D.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

namespace hdl_graph_slam {

class FloorDetectionNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FloorDetectionNodelet(const rclcpp::NodeOptions options = rclcpp::NodeOptions()) : Node("floor_detection_nodelet", options) {
    RCLCPP_DEBUG(this->get_logger(), "initializing floor_detection_nodelet...");

    initialize_params();

    points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic, 256, std::bind(&FloorDetectionNodelet::cloud_callback, this, std::placeholders::_1));
    floor_pub = this->create_publisher<hdl_graph_slam::msg::FloorCoeffs>("/floor_detection/floor_coeffs", 32);

    read_until_pub = this->create_publisher<std_msgs::msg::Header>("/floor_detection/read_until", 32);
    floor_filtered_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_filtered_points", 32);
    floor_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_points", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    tilt_deg = this->declare_parameter<double>("tilt_deg", 0.0);                           // approximate sensor tilt angle [deg]
    
    sensor_height = this->declare_parameter<double>("sensor_height", 2.0);                 // approximate sensor height [m]
    height_clip_range = this->declare_parameter<double>("height_clip_range", 1.0);         // points with heights in [sensor_height - height_clip_range, sensor_height + height_clip_range] will be used for floor detection
    floor_pts_thresh = this->declare_parameter<int>("floor_pts_thresh", 512);              // minimum number of support points of RANSAC to accept a detected floor plane
    floor_normal_thresh = this->declare_parameter<double>("floor_normal_thresh", 10.0);    // verticality check thresold for the detected floor plane [deg]
    use_normal_filtering = this->declare_parameter<bool>("use_normal_filtering", true);    // if true, points with "non-"vertical normals will be filtered before RANSAC
    normal_filter_thresh = this->declare_parameter<double>("normal_filter_thresh", 20.0);  // "non-"verticality check threshold [deg]

    points_topic = this->declare_parameter<std::string>("points_topic", "/velodyne_points");
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if(cloud->empty()) {
      return;
    }

    // floor detection
    // boost::optional<Eigen::Vector4f> floor = detect(cloud);
    std::optional<Eigen::Vector4f> floor = detect(cloud);

    // publish the detected floor coefficients
    hdl_graph_slam::msg::FloorCoeffs coeffs;
    coeffs.header = cloud_msg->header;
    if(floor) {
      coeffs.coeffs.resize(4);
      for(int i = 0; i < 4; i++) {
        coeffs.coeffs[i] = (*floor)[i];
      }
    }

    floor_pub->publish(coeffs);

    // for offline estimation
    std_msgs::msg::Header read_until;
    read_until.frame_id = points_topic;
    read_until.stamp.sec = cloud_msg->header.stamp.sec + 1.0;
    read_until_pub->publish(read_until);

    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);
  }

  /**
   * @brief detect the floor plane from a point cloud
   * @param cloud  input cloud
   * @return detected floor plane coefficients
   */
  std::optional<Eigen::Vector4f> detect(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    // compensate the tilt rotation
    Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();
    tilt_matrix.topLeftCorner(3, 3) = Eigen::AngleAxisf(tilt_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();

    // filtering before RANSAC (height and normal filtering)
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *filtered, tilt_matrix);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height + height_clip_range), false);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height - height_clip_range), true);

    if(use_normal_filtering) {
      filtered = normal_filtering(filtered);
    }

    pcl::transformPointCloud(*filtered, *filtered, static_cast<Eigen::Matrix4f>(tilt_matrix.inverse()));

    if(floor_filtered_pub->get_subscription_count()) {
      filtered->header = cloud->header;
      sensor_msgs::msg::PointCloud2 pc2;
      pcl::toROSMsg(*filtered, pc2);
      floor_filtered_pub->publish(pc2);
    }

    // too few points for RANSAC
    if(filtered->size() < floor_pts_thresh) {
      return std::nullopt;
    }

    // RANSAC
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(filtered));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(0.1);
    ransac.computeModel();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // too few inliers
    if(inliers->indices.size() < floor_pts_thresh) {
      return std::nullopt;
    }

    // verticality check of the detected floor's normal
    Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();

    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);

    double dot = coeffs.head<3>().dot(reference.head<3>());
    if(std::abs(dot) < std::cos(floor_normal_thresh * M_PI / 180.0)) {
      // the normal is not vertical
      return std::nullopt;
    }

    // make the normal upward
    if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
      coeffs *= -1.0f;
    }

    if(floor_points_pub->get_subscription_count()) {
      pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);
      inlier_cloud->header = cloud->header;

      sensor_msgs::msg::PointCloud2 pc2;
      pcl::toROSMsg(*inlier_cloud, pc2);
      floor_points_pub->publish(pc2);
    }

    return Eigen::Vector4f(coeffs);
  }

  /**
   * @brief plane_clip
   * @param src_cloud
   * @param plane
   * @param negative
   * @return
   */
  pcl::PointCloud<PointT>::Ptr plane_clip(const pcl::PointCloud<PointT>::Ptr& src_cloud, const Eigen::Vector4f& plane, bool negative) const {
    pcl::PlaneClipper3D<PointT> clipper(plane);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);

    clipper.clipPointCloud3D(*src_cloud, indices->indices);

    pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(src_cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*dst_cloud);

    return dst_cloud;
  }

  /**
   * @brief filter points with non-vertical normals
   * @param cloud  input cloud
   * @return filtered cloud
   */
  pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(10);
    ne.setViewPoint(0.0f, 0.0f, sensor_height);
    ne.compute(*normals);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());

    for(size_t i = 0; i < cloud->size(); i++) {
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
      if(std::abs(dot) > std::cos(normal_filter_thresh * M_PI / 180.0)) {
        filtered->push_back(cloud->at(i));
      }
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    return filtered;
  }

private:
  // ROS topics
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;

  rclcpp::Publisher<hdl_graph_slam::msg::FloorCoeffs>::SharedPtr floor_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_points_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_filtered_pub;

  std::string points_topic;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;

  // floor detection parameters
  // see initialize_params() for the details
  double tilt_deg;
  double sensor_height;
  double height_clip_range;

  size_t floor_pts_thresh;
  double floor_normal_thresh;

  bool use_normal_filtering;
  double normal_filter_thresh;
};

}  // namespace hdl_graph_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_graph_slam::FloorDetectionNodelet)

