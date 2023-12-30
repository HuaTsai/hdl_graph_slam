// SPDX-License-Identifier: BSD-2-Clause
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
// #include <pcl_ros/point_cloud.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace hdl_graph_slam {

class PrefilteringNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  PrefilteringNodelet(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("prefiltering_nodelet", options) {}

  void onInit() {
    initialize_params();

    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    bool deskewing = this->declare_parameter<bool>("deskewing", false);
    if(deskewing) {
      imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("/imu/data", 1, std::bind(&PrefilteringNodelet::imu_callback, this, std::placeholders::_1));
    }

    points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("/velodyne_points", 64, std::bind(&PrefilteringNodelet::cloud_callback, this, std::placeholders::_1));
    points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", 32);
    colored_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/colored_points", 32);
  }

private:
  void initialize_params() {
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
    }

    std::string outlier_removal_method = this->declare_parameter<std::string>("outlier_removal_method", "STATISTICAL");
    if(outlier_removal_method == "STATISTICAL") {
      int mean_k = this->declare_parameter<int>("statistical_mean_k", 20);
      double stddev_mul_thresh = this->declare_parameter<double>("statistical_stddev", 1.0);
      std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;
    } else if(outlier_removal_method == "RADIUS") {
      double radius = this->declare_parameter<double>("radius_radius", 0.8);
      int min_neighbors = this->declare_parameter<int>("radius_min_neighbors", 2);
      std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;
    } else {
      std::cout << "outlier_removal: NONE" << std::endl;
    }

    use_distance_filter = this->declare_parameter<bool>("use_distance_filter", true);
    distance_near_thresh = this->declare_parameter<double>("distance_near_thresh", 1.0);
    distance_far_thresh = this->declare_parameter<double>("distance_far_thresh", 100.0);

    base_link_frame = this->declare_parameter<std::string>("base_link_frame", "");
    scan_period = this->declare_parameter<double>("scan_period", 0.1);
  }

  void imu_callback(sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    imu_queue.push_back(imu_msg);
  }

  void cloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr src_cloud_r) {
    pcl::PointCloud<PointT>::Ptr src_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*src_cloud_r, *src_cloud);
    if(src_cloud->empty()) {
      return;
    }

    src_cloud = deskewing(src_cloud);

    // if base_link_frame is defined, transform the input cloud to the frame
    if(!base_link_frame.empty()) {
      geometry_msgs::msg::TransformStamped transform;
      if (!tf_buffer->canTransform(base_link_frame, src_cloud->header.frame_id, rclcpp::Time(0))) {
        RCLCPP_INFO(this->get_logger(), "Failed to find transform between %s and %s", base_link_frame.c_str(), src_cloud->header.frame_id.c_str());
      } else {
        try {
          transform = tf_buffer->lookupTransform(base_link_frame, src_cloud->header.frame_id, rclcpp::Time(0));
        } catch (const tf2::TransformException& e) {
          RCLCPP_INFO(this->get_logger(), "Failed to find transform between %s and %s", base_link_frame.c_str(), src_cloud->header.frame_id.c_str());
        }
      }

      pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
      pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
      transformed->header.frame_id = base_link_frame;
      transformed->header.stamp = src_cloud->header.stamp;
      src_cloud = transformed;
    }

    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);
    filtered = downsample(filtered);
    filtered = outlier_removal(filtered);

    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(*filtered, pc2);
    points_pub->publish(pc2);
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());

    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();
      return d > distance_near_thresh && d < distance_far_thresh;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::Ptr deskewing(pcl::PointCloud<PointT>::Ptr cloud) {
    rclcpp::Time stamp = pcl_conversions::fromPCL(cloud->header.stamp);
    if(imu_queue.empty()) {
      return cloud;
    }

    // the color encodes the point number in the point sequence
    if(colored_pub->get_subscription_count()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
      colored->header = cloud->header;
      colored->is_dense = cloud->is_dense;
      colored->width = cloud->width;
      colored->height = cloud->height;
      colored->resize(cloud->size());

      for(size_t i = 0; i < cloud->size(); i++) {
        double t = static_cast<double>(i) / cloud->size();
        colored->at(i).getVector4fMap() = cloud->at(i).getVector4fMap();
        colored->at(i).r = 255 * t;
        colored->at(i).g = 128;
        colored->at(i).b = 255 * (1 - t);
      }
      sensor_msgs::msg::PointCloud2 pc2;
      pcl::toROSMsg(*colored, pc2);
      colored_pub->publish(pc2);
    }

    sensor_msgs::msg::Imu::ConstSharedPtr imu_msg = imu_queue.front();

    auto loc = imu_queue.begin();
    for(; loc != imu_queue.end(); loc++) {
      imu_msg = (*loc);
      if(rclcpp::Time((*loc)->header.stamp) > stamp) {
        break;
      }
    }

    imu_queue.erase(imu_queue.begin(), loc);

    Eigen::Vector3f ang_v(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    ang_v *= -1;

    pcl::PointCloud<PointT>::Ptr deskewed(new pcl::PointCloud<PointT>());
    deskewed->header = cloud->header;
    deskewed->is_dense = cloud->is_dense;
    deskewed->width = cloud->width;
    deskewed->height = cloud->height;
    deskewed->resize(cloud->size());

    for(size_t i = 0; i < cloud->size(); i++) {
      const auto& pt = cloud->at(i);

      // TODO: transform IMU data into the LIDAR frame
      double delta_t = scan_period * static_cast<double>(i) / cloud->size();
      Eigen::Quaternionf delta_q(1, delta_t / 2.0 * ang_v[0], delta_t / 2.0 * ang_v[1], delta_t / 2.0 * ang_v[2]);
      Eigen::Vector3f pt_ = delta_q.inverse() * pt.getVector3fMap();

      deskewed->at(i) = cloud->at(i);
      deskewed->at(i).getVector3fMap() = pt_;
    }

    return deskewed;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_queue;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_pub;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;

  std::string base_link_frame;

  bool use_distance_filter;
  double distance_near_thresh;
  double distance_far_thresh;
  double scan_period;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;
};

}  // namespace hdl_graph_slam

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hdl_graph_slam::PrefilteringNodelet>();
  node->onInit();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
