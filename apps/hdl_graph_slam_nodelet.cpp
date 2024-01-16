// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <iostream>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nmea_msgs/msg/sentence.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geographic_msgs/msg/geo_point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <hdl_graph_slam/msg/floor_coeffs.hpp>

#include <hdl_graph_slam/srv/save_map.hpp>
#include <hdl_graph_slam/srv/load_graph.hpp>
#include <hdl_graph_slam/srv/dump_graph.hpp>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/nmea_sentence_parser.hpp>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

namespace hdl_graph_slam {

class HdlGraphSlamNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;

  HdlGraphSlamNodelet(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("hdl_graph_slam", options) {}
  virtual ~HdlGraphSlamNodelet() {}

  void onInit() {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // init parameters
    published_odom_topic = this->declare_parameter<std::string>("published_odom_topic", "/odom");
    map_frame_id = this->declare_parameter<std::string>("map_frame_id", "map");
    odom_frame_id = this->declare_parameter<std::string>("odom_frame_id", "odom");
    map_cloud_resolution = this->declare_parameter<double>("map_cloud_resolution", 0.05);
    trans_odom2map.setIdentity();

    max_keyframes_per_update = this->declare_parameter<int>("max_keyframes_per_update", 10);
    std::string g2o_solver_type = this->declare_parameter<std::string>("g2o_solver_type", "lm_var");

    fix_first_node = this->declare_parameter<bool>("fix_first_node", false);
    fix_first_node_stddev = this->declare_parameter<std::string>("fix_first_node_stddev", "1 1 1 1 1 1");
    odometry_edge_robust_kernel = this->declare_parameter<std::string>("odometry_edge_robust_kernel", "NONE");
    odometry_edge_robust_kernel_size = this->declare_parameter<double>("odometry_edge_robust_kernel_size", 1.0);

    gps_edge_robust_kernel = this->declare_parameter<std::string>("gps_edge_robust_kernel", "NONE");
    gps_edge_robust_kernel_size = this->declare_parameter<double>("gps_edge_robust_kernel_size", 1.0);

    imu_orientation_edge_robust_kernel = this->declare_parameter<std::string>("imu_orientation_edge_robust_kernel", "NONE");
    imu_orientation_edge_robust_kernel_size = this->declare_parameter<double>("imu_orientation_edge_robust_kernel_size", 1.0);
    imu_acceleration_edge_robust_kernel = this->declare_parameter<std::string>("imu_acceleration_edge_robust_kernel", "NONE");
    imu_acceleration_edge_robust_kernel_size = this->declare_parameter<double>("imu_acceleration_edge_robust_kernel_size", 1.0);
    floor_edge_robust_kernel = this->declare_parameter<std::string>("floor_edge_robust_kernel", "NONE");
    floor_edge_robust_kernel_size = this->declare_parameter<double>("floor_edge_robust_kernel_size", 1.0);
    loop_closure_edge_robust_kernel = this->declare_parameter<std::string>("loop_closure_edge_robust_kernel", "NONE");
    loop_closure_edge_robust_kernel_size = this->declare_parameter<double>("loop_closure_edge_robust_kernel_size", 1.0);
    fix_first_node_adaptive = this->declare_parameter<bool>("fix_first_node_adaptive", true);
    g2o_solver_num_iterations = this->declare_parameter<int>("g2o_solver_num_iterations", 1024);

    anchor_node = nullptr;
    anchor_edge = nullptr;
    floor_plane_node = nullptr;
    graph_slam.reset(new GraphSLAM(g2o_solver_type));
    keyframe_updater.reset(new KeyframeUpdater(this));
    loop_detector.reset(new LoopDetector(this));
    map_cloud_generator.reset(new MapCloudGenerator());
    inf_calclator.reset(new InformationMatrixCalculator(this));
    nmea_parser.reset(new NmeaSentenceParser());

    gps_time_offset = this->declare_parameter<double>("gps_time_offset", 0.0);
    gps_edge_stddev_xy = this->declare_parameter<double>("gps_edge_stddev_xy", 10000.0);
    gps_edge_stddev_z = this->declare_parameter<double>("gps_edge_stddev_z", 10.0);
    floor_edge_stddev = this->declare_parameter<double>("floor_edge_stddev", 10.0);

    imu_time_offset = this->declare_parameter<double>("imu_time_offset", 0.0);
    enable_imu_orientation = this->declare_parameter<bool>("enable_imu_orientation", false);
    enable_imu_acceleration = this->declare_parameter<bool>("enable_imu_acceleration", false);
    imu_orientation_edge_stddev = this->declare_parameter<double>("imu_orientation_edge_stddev", 0.1);
    imu_acceleration_edge_stddev = this->declare_parameter<double>("imu_acceleration_edge_stddev", 3.0);

    points_topic = this->declare_parameter<std::string>("points_topic", "/velodyne_points");

    // subscribers
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::msg::Odometry>(this, published_odom_topic, rclcpp::QoS(256).get_rmw_qos_profile()));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::msg::PointCloud2>(this, "/filtered_points", rclcpp::QoS(32).get_rmw_qos_profile()));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub));
    sync->registerCallback(std::bind(&HdlGraphSlamNodelet::cloud_callback, this, std::placeholders::_1, std::placeholders::_2));
    imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("gpsimu_driver/imu_data", 1024, std::bind(&HdlGraphSlamNodelet::imu_callback, this, std::placeholders::_1));
    floor_sub = this->create_subscription<hdl_graph_slam::msg::FloorCoeffs>("floor_detection/floor_coeffs", 1024, std::bind(&HdlGraphSlamNodelet::floor_coeffs_callback, this, std::placeholders::_1));

    bool enable_gps = this->declare_parameter<bool>("enable_gps", true);
    if(enable_gps) {
      gps_sub = this->create_subscription<geographic_msgs::msg::GeoPointStamped>("gps/geopoint", 1024, std::bind(&HdlGraphSlamNodelet::gps_callback, this, std::placeholders::_1));
      nmea_sub = this->create_subscription<nmea_msgs::msg::Sentence>("gpsimu_driver/nmea_sentence", 1024, std::bind(&HdlGraphSlamNodelet::nmea_callback, this, std::placeholders::_1));
      navsat_sub = this->create_subscription<sensor_msgs::msg::NavSatFix>("gps/navsat", 1024, std::bind(&HdlGraphSlamNodelet::navsat_callback, this, std::placeholders::_1));
    }

    // publishers
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();
    markers_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("hdl_graph_slam/markers", 16);
    // odom2map_pub = this->create_publisher<geometry_msgs::msg::TransformStamped>("hdl_graph_slam/odom2pub", 16);
    map_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("hdl_graph_slam/map_points", qos);
    read_until_pub = this->create_publisher<std_msgs::msg::Header>("hdl_graph_slam/read_until", 32);


    // FIXME: mt_nh?
    load_service_server = this->create_service<hdl_graph_slam::srv::LoadGraph>("hdl_graph_slam/load", std::bind(&HdlGraphSlamNodelet::load_service, this, std::placeholders::_1, std::placeholders::_2));
    dump_service_server = this->create_service<hdl_graph_slam::srv::DumpGraph>("hdl_graph_slam/dump", std::bind(&HdlGraphSlamNodelet::dump_service, this, std::placeholders::_1, std::placeholders::_2));
    save_map_service_server = this->create_service<hdl_graph_slam::srv::SaveMap>("hdl_graph_slam/save_map", std::bind(&HdlGraphSlamNodelet::save_map_service, this, std::placeholders::_1, std::placeholders::_2));

    graph_updated = false;
    double graph_update_interval = this->declare_parameter<double>("graph_update_interval", 3.0);
    double map_cloud_update_interval = this->declare_parameter<double>("map_cloud_update_interval", 10.0);
    optimization_timer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(graph_update_interval * 1000)), std::bind(&HdlGraphSlamNodelet::optimization_timer_callback, this));
    map_publish_timer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(map_cloud_update_interval * 1000)), std::bind(&HdlGraphSlamNodelet::map_points_publish_timer_callback, this));
    // this->create_wall_timer()
  }

private:
  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(nav_msgs::msg::Odometry::ConstSharedPtr odom_msg, sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    const rclcpp::Time &stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom;
    tf2::fromMsg(odom_msg->pose.pose, odom);

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(base_frame_id.empty()) {
      base_frame_id = cloud_msg->header.frame_id;
    }

    if(!keyframe_updater->update(odom)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
        std_msgs::msg::Header read_until;
        read_until.stamp = stamp + rclcpp::Duration(10, 0);
        read_until.frame_id = points_topic;
        read_until_pub->publish(read_until);
        read_until.frame_id = "/filtered_points";
        read_until_pub->publish(read_until);
      }

      return;
    }

    double accum_d = keyframe_updater->get_accum_distance();
    KeyFrame::Ptr keyframe(new KeyFrame(stamp, odom, accum_d, cloud, this));

    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);
  }

  /**
   * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);

    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    std::cout << "flush_keyframe_queue - keyframes len:"<< keyframes.size() << std::endl;
    int num_processed = 0;
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframes will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry3d odom = odom2map * keyframe->odom;
      keyframe->node = graph_slam->add_se3_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        
        if(this->get_parameter("fix_first_node").as_bool()) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(this->get_parameter("fix_first_node_stddev").as_string());
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }

          anchor_node = graph_slam->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }

      if(i == 0 && keyframes.empty()) {
        continue;
      }

      // add edge between consecutive keyframes
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

      Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      odometry_edge_robust_kernel = this->get_parameter("odometry_edge_robust_kernel").as_string();
      odometry_edge_robust_kernel_size = this->get_parameter("odometry_edge_robust_kernel_size").as_double();
      graph_slam->add_robust_kernel(edge, odometry_edge_robust_kernel, odometry_edge_robust_kernel_size);
    }

    std_msgs::msg::Header read_until;
    read_until.stamp = keyframe_queue[num_processed]->stamp + rclcpp::Duration(10, 0);
    read_until.frame_id = points_topic;
    read_until_pub->publish(read_until);
    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }

  void nmea_callback(nmea_msgs::msg::Sentence::ConstSharedPtr nmea_msg) {
    GPRMC grmc = nmea_parser->parse(nmea_msg->sentence);

    if(grmc.status != 'A') {
      return;
    }

    auto gps_msg = std::make_shared<geographic_msgs::msg::GeoPointStamped>();
    gps_msg->header = nmea_msg->header;
    gps_msg->position.latitude = grmc.latitude;
    gps_msg->position.longitude = grmc.longitude;
    gps_msg->position.altitude = NAN;

    gps_callback(gps_msg);
  }

  void navsat_callback(sensor_msgs::msg::NavSatFix::ConstSharedPtr navsat_msg) {
    auto gps_msg = std::make_shared<geographic_msgs::msg::GeoPointStamped>();
    gps_msg->header = navsat_msg->header;
    gps_msg->position.latitude = navsat_msg->latitude;
    gps_msg->position.longitude = navsat_msg->longitude;
    gps_msg->position.altitude = navsat_msg->altitude;
    gps_callback(gps_msg);
  }

  /**
   * @brief received gps data is added to #gps_queue
   * @param gps_msg
   */
  void gps_callback(geographic_msgs::msg::GeoPointStamped::SharedPtr gps_msg) {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);
    // gps_msg->header.stamp = gps_msg->header.stamp + rclcpp::Duration(gps_time_offset);
    // gps_msg->header.stamp = gps_msg->header.stamp + rclcpp::Duration::from_seconds(gps_time_offset);
    gps_msg->header.stamp = rclcpp::Time(gps_msg->header.stamp) + rclcpp::Duration::from_seconds(gps_time_offset);
    gps_queue.push_back(gps_msg);
  }

  /**
   * @brief
   * @return
   */
  bool flush_gps_queue() {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);

    if(keyframes.empty() || gps_queue.empty()) {
      return false;
    }

    bool updated = false;
    auto gps_cursor = gps_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > gps_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*gps_cursor)->header.stamp || keyframe->utm_coord) {
        continue;
      }

      // find the gps data which is closest to the keyframe
      auto closest_gps = gps_cursor;
      for(auto gps = gps_cursor; gps != gps_queue.end(); gps++) {
        auto dt = (rclcpp::Time((*closest_gps)->header.stamp) - keyframe->stamp).seconds();
        auto dt2 = (rclcpp::Time((*gps)->header.stamp) - keyframe->stamp).seconds();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_gps = gps;
      }

      // if the time residual between the gps and keyframe is too large, skip it
      gps_cursor = closest_gps;
      if(0.2 < std::abs((rclcpp::Time((*closest_gps)->header.stamp) - keyframe->stamp).seconds())) {
        continue;
      }

      // convert (latitude, longitude, altitude) -> (easting, northing, altitude) in UTM coordinate
      geodesy::UTMPoint utm;
      geodesy::fromMsg((*closest_gps)->position, utm);
      Eigen::Vector3d xyz(utm.easting, utm.northing, utm.altitude);

      // the first gps data position will be the origin of the map
      if(!zero_utm) {
        zero_utm = xyz;
      }
      xyz -= (*zero_utm);

      keyframe->utm_coord = xyz;

      g2o::OptimizableGraph::Edge* edge;
      if(std::isnan(xyz.z())) {
        Eigen::Matrix2d information_matrix = Eigen::Matrix2d::Identity() / gps_edge_stddev_xy;
        edge = graph_slam->add_se3_prior_xy_edge(keyframe->node, xyz.head<2>(), information_matrix);
      } else {
        Eigen::Matrix3d information_matrix = Eigen::Matrix3d::Identity();
        information_matrix.block<2, 2>(0, 0) /= gps_edge_stddev_xy;
        information_matrix(2, 2) /= gps_edge_stddev_z;
        edge = graph_slam->add_se3_prior_xyz_edge(keyframe->node, xyz, information_matrix);
      }
      graph_slam->add_robust_kernel(edge, gps_edge_robust_kernel, gps_edge_robust_kernel_size);

      updated = true;
    }

    auto remove_loc = std::upper_bound(gps_queue.begin(), gps_queue.end(), keyframes.back()->stamp, [=](const rclcpp::Time& stamp, geographic_msgs::msg::GeoPointStamped::ConstSharedPtr geopoint) { return stamp < geopoint->header.stamp; });
    gps_queue.erase(gps_queue.begin(), remove_loc);
    return updated;
  }

  void imu_callback(sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    if(!enable_imu_orientation && !enable_imu_acceleration) {
      return;
    }

    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    imu_msg->header.stamp = rclcpp::Duration::from_seconds(imu_time_offset) + imu_msg->header.stamp;
    imu_queue.push_back(imu_msg);
  }

  bool flush_imu_queue() {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(keyframes.empty() || imu_queue.empty() || base_frame_id.empty()) {
      return false;
    }

    bool updated = false;
    auto imu_cursor = imu_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > imu_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*imu_cursor)->header.stamp || keyframe->acceleration) {
        continue;
      }

      // find imu data which is closest to the keyframe
      auto closest_imu = imu_cursor;
      for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
        auto dt = (rclcpp::Time((*closest_imu)->header.stamp) - keyframe->stamp).seconds();
        auto dt2 = (rclcpp::Time((*imu)->header.stamp) - keyframe->stamp).seconds();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_imu = imu;
      }

      imu_cursor = closest_imu;
      if(0.2 < std::abs((rclcpp::Time((*closest_imu)->header.stamp) - keyframe->stamp).seconds())) {
        continue;
      }

      // const auto& imu_ori = (*closest_imu)->orientation;
      // const auto& imu_acc = (*closest_imu)->linear_acceleration;

      geometry_msgs::msg::Vector3Stamped acc_imu;
      geometry_msgs::msg::Vector3Stamped acc_base;
      geometry_msgs::msg::QuaternionStamped quat_imu;
      geometry_msgs::msg::QuaternionStamped quat_base;

      quat_imu.header.frame_id = acc_imu.header.frame_id = (*closest_imu)->header.frame_id;
      quat_imu.header.stamp = acc_imu.header.stamp = rclcpp::Time(0);
      acc_imu.vector = (*closest_imu)->linear_acceleration;
      quat_imu.quaternion = (*closest_imu)->orientation;

      try {
        tf_buffer->transform(acc_imu, acc_base, base_frame_id);
        tf_buffer->transform(quat_imu, quat_base, base_frame_id);
      } catch(tf2::TransformException &e) {
        std::cerr << "failed to find transform!!" << std::endl;
        return false;
      }

      keyframe->acceleration = Eigen::Vector3d(acc_base.vector.x, acc_base.vector.y, acc_base.vector.z);
      keyframe->orientation = Eigen::Quaterniond(quat_base.quaternion.w, quat_base.quaternion.x, quat_base.quaternion.y, quat_base.quaternion.z);
      keyframe->orientation = keyframe->orientation;
      if(keyframe->orientation->w() < 0.0) {
        keyframe->orientation->coeffs() = -keyframe->orientation->coeffs();
      }

      if(enable_imu_orientation) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / imu_orientation_edge_stddev;
        auto edge = graph_slam->add_se3_prior_quat_edge(keyframe->node, *keyframe->orientation, info);
        graph_slam->add_robust_kernel(edge, imu_orientation_edge_robust_kernel, imu_orientation_edge_robust_kernel_size);
      }

      if(enable_imu_acceleration) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / imu_acceleration_edge_stddev;
        g2o::OptimizableGraph::Edge* edge = graph_slam->add_se3_prior_vec_edge(keyframe->node, -Eigen::Vector3d::UnitZ(), *keyframe->acceleration, info);
        graph_slam->add_robust_kernel(edge, imu_acceleration_edge_robust_kernel, imu_acceleration_edge_robust_kernel_size);
      }
      updated = true;
    }

    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const rclcpp::Time& stamp, sensor_msgs::msg::Imu::ConstSharedPtr imu) { return stamp < imu->header.stamp; });
    imu_queue.erase(imu_queue.begin(), remove_loc);

    return updated;
  }

  /**
   * @brief received floor coefficients are added to #floor_coeffs_queue
   * @param floor_coeffs_msg
   */
  void floor_coeffs_callback(hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr floor_coeffs_msg) {
    if(floor_coeffs_msg->coeffs.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);
    floor_coeffs_queue.push_back(floor_coeffs_msg);
  }

  /**
   * @brief this methods associates floor coefficients messages with registered keyframes, and then adds the associated coeffs to the pose graph
   * @return if true, at least one floor plane edge is added to the pose graph
   */
  bool flush_floor_queue() {
    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);

    if(keyframes.empty()) {
      return false;
    }

    const auto& latest_keyframe_stamp = keyframes.back()->stamp;

    bool updated = false;
    for(const auto& floor_coeffs : floor_coeffs_queue) {
      if(rclcpp::Time(floor_coeffs->header.stamp) > latest_keyframe_stamp) {
        break;
      }

      auto found = keyframe_hash.find(floor_coeffs->header.stamp);
      if(found == keyframe_hash.end()) {
        continue;
      }

      if(!floor_plane_node) {
        floor_plane_node = graph_slam->add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0));
        floor_plane_node->setFixed(true);
      }

      const auto& keyframe = found->second;

      Eigen::Vector4d coeffs(floor_coeffs->coeffs[0], floor_coeffs->coeffs[1], floor_coeffs->coeffs[2], floor_coeffs->coeffs[3]);
      Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * (1.0 / floor_edge_stddev);
      auto edge = graph_slam->add_se3_plane_edge(keyframe->node, floor_plane_node, coeffs, information);
      graph_slam->add_robust_kernel(edge, floor_edge_robust_kernel, floor_edge_robust_kernel_size);

      keyframe->floor_coeffs = coeffs;

      updated = true;
    }

    auto remove_loc = std::upper_bound(floor_coeffs_queue.begin(), floor_coeffs_queue.end(), latest_keyframe_stamp, [=](const rclcpp::Time& stamp, hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr coeffs) { return stamp < coeffs->header.stamp; });
    floor_coeffs_queue.erase(floor_coeffs_queue.begin(), remove_loc);

    return updated;
  }

  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback() {
    if(!map_points_pub->get_subscription_count() || !graph_updated) {
      return;
    }

    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, map_cloud_resolution);
    if(!cloud) {
      return;
    }

    cloud->header.frame_id = map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::msg::PointCloud2 pc2;
    pcl::toROSMsg(*cloud, pc2);

    map_points_pub->publish(pc2);
  }

  /**
   * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback() {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes and floor coeffs in the queues to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
      std_msgs::msg::Header read_until;
      read_until.stamp = now() + rclcpp::Duration(30, 0);
      read_until.frame_id = points_topic;
      read_until_pub->publish(read_until);
      read_until.frame_id = "/filtered_points";
      read_until_pub->publish(read_until);
    }

    if(!keyframe_updated && !flush_floor_queue() && !flush_gps_queue() && !flush_imu_queue()) {
      return;
    }

    // loop detection
    std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes);
    for(const auto& loop : loops) {
      Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
      Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(loop->key1->cloud, loop->key2->cloud, relpose);
      auto edge = graph_slam->add_se3_edge(loop->key1->node, loop->key2->node, relpose, information_matrix);
      graph_slam->add_robust_kernel(edge, loop_closure_edge_robust_kernel, loop_closure_edge_robust_kernel_size);
    }

    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    // move the first node anchor position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && fix_first_node_adaptive) {
      Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3*>(anchor_edge->vertices()[1])->estimate();
      anchor_node->setEstimate(anchor_target);
    }

    // optimize the pose graph
    graph_slam->optimize(g2o_solver_num_iterations);

    // publish tf
    const auto& keyframe = keyframes.back();
    Eigen::Isometry3d trans = keyframe->node->estimate() * keyframe->odom.inverse();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix().cast<float>();
    trans_odom2map_mutex.unlock();

    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    // if(odom2map_pub->get_subscription_count()) {
    geometry_msgs::msg::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix().cast<float>(), map_frame_id, odom_frame_id);
    tf_broadcaster->sendTransform(ts);
    // }

    if(markers_pub->get_subscription_count()) {
      auto markers = create_marker_array(now());
      markers_pub->publish(markers);
    }
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::msg::MarkerArray create_marker_array(const rclcpp::Time& stamp) const {
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.resize(4);

    // node markers
    visualization_msgs::msg::Marker& traj_marker = markers.markers[0];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

    visualization_msgs::msg::Marker& imu_marker = markers.markers[1];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(size_t i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 1.0 - p;
      traj_marker.colors[i].g = p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::msg::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::msg::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.1;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

    // edge markers
    visualization_msgs::msg::Marker& edge_marker = markers.markers[2];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 1.0 - p1;
        edge_marker.colors[i * 2].g = p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          edge_marker.points[i * 2].z += 0.5;
          edge_marker.points[i * 2 + 1].z += 0.5;
        }

        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    // sphere
    visualization_msgs::msg::Marker& sphere_marker = markers.markers[3];
    sphere_marker.header.frame_id = "map";
    sphere_marker.header.stamp = stamp;
    sphere_marker.ns = "loop_close_radius";
    sphere_marker.id = 3;
    sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;

    if(!keyframes.empty()) {
      Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
      sphere_marker.pose.position.x = pos.x();
      sphere_marker.pose.position.y = pos.y();
      sphere_marker.pose.position.z = pos.z();
    }
    sphere_marker.pose.orientation.w = 1.0;
    sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

    sphere_marker.color.r = 1.0;
    sphere_marker.color.a = 0.3;

    return markers;
  }


  /**
   * @brief load all data from a directory
   * @param req
   * @param res
   * @return
   */
  bool load_service(hdl_graph_slam::srv::LoadGraph::Request::ConstSharedPtr req,
                    hdl_graph_slam::srv::LoadGraph::Response::SharedPtr res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req->path;

    std::cout << "loading data from:" << directory << std::endl;

    // Load graph.
    graph_slam->load(directory + "/graph.g2o");
    
    // Iterate over the items in this directory and count how many sub directories there are. 
    // This will give an upper limit on how many keyframe indexes we can expect to find.
    boost::filesystem::directory_iterator begin(directory), end;
    int max_directory_count = std::count_if(begin, end,
        [](const boost::filesystem::directory_entry & d) {
            return boost::filesystem::is_directory(d.path()); // only return true if a direcotry
    });

    // Load keyframes by looping through key frame indexes that we expect to see.
    for(int i = 0; i < max_directory_count; i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;
      std::string key_frame_directory = sst.str();

      // If key_frame_directory doesnt exist, then we have run out so lets stop looking.
      if(!boost::filesystem::is_directory(key_frame_directory)) {
        break;
      }

      KeyFrame::Ptr keyframe(new KeyFrame(key_frame_directory, graph_slam->graph.get(), this));
      keyframes.push_back(keyframe);
    }
    std::cout << "loaded " << keyframes.size() << " keyframes" <<std::endl;
    
    // Load special nodes.
    std::ifstream ifs(directory + "/special_nodes.csv");
    if(!ifs) {
      return false;
    }
    while(!ifs.eof()) {
      std::string token;
      ifs >> token;
      if(token == "anchor_node") {
        int id = 0;
        ifs >> id;
        anchor_node = static_cast<g2o::VertexSE3*>(graph_slam->graph->vertex(id));
      } else if(token == "anchor_edge") {
        int id = 0;
        ifs >> id; 
        // We have no way of directly pulling the edge based on the edge ID that we have just read in.
        // Fortunatly anchor edges are always attached to the anchor node so we can iterate over 
        // the edges that listed against the node untill we find the one that matches our ID.
        if(anchor_node){
          auto edges = anchor_node->edges();

          for(auto e : edges) {
            int edgeID =  e->id();
            if (edgeID == id){
              anchor_edge = static_cast<g2o::EdgeSE3*>(e);

              break;
            }
          } 
        }
      } else if(token == "floor_node") {
        int id = 0;
        ifs >> id;
        floor_plane_node = static_cast<g2o::VertexPlane*>(graph_slam->graph->vertex(id));
      }
    }

    // check if we have any non null special nodes, if all are null then lets not bother.
    if(anchor_node->id() || anchor_edge->id() || floor_plane_node->id()) {
      std::cout << "loaded special nodes - ";

      // check exists before printing information about each special node
      if(anchor_node->id()) {
        std::cout << " anchor_node: " << anchor_node->id();
      }
      if(anchor_edge->id()) {
        std::cout << " anchor_edge: " << anchor_edge->id();
      }
      if(floor_plane_node->id()) {
        std::cout << " floor_node: " << floor_plane_node->id();
      }
      
      // finish with a new line
      std::cout << std::endl;
    }

    // Update our keyframe snapshot so we can publish a map update, trigger update with graph_updated = true.
    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());

    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;

    res->success = true;

    std::cout << "snapshot updated" << std::endl << "loading successful" <<std::endl;

    return true;
  }


  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(hdl_graph_slam::srv::DumpGraph::Request::ConstSharedPtr req,
                    hdl_graph_slam::srv::DumpGraph::Response::SharedPtr res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req->destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "dumping data to:" << directory << std::endl;
    // save graph 
    graph_slam->save(directory + "/graph.g2o");

    // save keyframes
    for(size_t i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }

    if(zero_utm) {
      std::ofstream zero_utm_ofs(directory + "/zero_utm");
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

    res->success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(hdl_graph_slam::srv::SaveMap::Request::ConstSharedPtr req,
                        hdl_graph_slam::srv::SaveMap::Response::SharedPtr res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req->resolution);
    if(!cloud) {
      res->success = false;
      return true;
    }

    if(zero_utm && req->utm) {
      for(auto& pt : cloud->points) {
        pt.getVector3fMap() += (*zero_utm).cast<float>();
      }
    }

    cloud->header.frame_id = map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    if(zero_utm) {
      std::ofstream ofs(req->destination + ".utm");
      ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    int ret = pcl::io::savePCDFileBinary(req->destination, *cloud);
    res->success = ret == 0;

    return true;
  }

private:
  // ROS
  rclcpp::TimerBase::SharedPtr optimization_timer;
  rclcpp::TimerBase::SharedPtr map_publish_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  rclcpp::Subscription<geographic_msgs::msg::GeoPointStamped>::SharedPtr gps_sub;
  rclcpp::Subscription<nmea_msgs::msg::Sentence>::SharedPtr nmea_sub;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr navsat_sub;

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr navsat_msg;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<hdl_graph_slam::msg::FloorCoeffs>::SharedPtr floor_sub;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub;

  std::string published_odom_topic;
  std::string map_frame_id;
  std::string odom_frame_id;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix4f trans_odom2map;
  // rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr odom2map_pub;

  std::string points_topic;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  rclcpp::Service<hdl_graph_slam::srv::LoadGraph>::SharedPtr load_service_server;
  rclcpp::Service<hdl_graph_slam::srv::DumpGraph>::SharedPtr dump_service_server;
  rclcpp::Service<hdl_graph_slam::srv::SaveMap>::SharedPtr save_map_service_server;

  // keyframe queue
  std::string base_frame_id;
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;
  bool fix_first_node;
  std::string fix_first_node_stddev;
  std::string odometry_edge_robust_kernel;
  double odometry_edge_robust_kernel_size;

  // gps queue
  double gps_time_offset;
  double gps_edge_stddev_xy;
  double gps_edge_stddev_z;
  boost::optional<Eigen::Vector3d> zero_utm;
  std::mutex gps_queue_mutex;
  std::deque<geographic_msgs::msg::GeoPointStamped::ConstSharedPtr> gps_queue;
  std::string gps_edge_robust_kernel;
  double gps_edge_robust_kernel_size;

  // imu queue
  double imu_time_offset;
  bool enable_imu_orientation;
  double imu_orientation_edge_stddev;
  bool enable_imu_acceleration;
  double imu_acceleration_edge_stddev;
  std::mutex imu_queue_mutex;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_queue;

  // floor_coeffs queue
  double floor_edge_stddev;
  std::mutex floor_coeffs_queue_mutex;
  std::deque<hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr> floor_coeffs_queue;

  // for map cloud generation
  std::atomic_bool graph_updated;
  double map_cloud_resolution;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  std::string imu_orientation_edge_robust_kernel;
  double imu_orientation_edge_robust_kernel_size;
  std::string imu_acceleration_edge_robust_kernel;
  double imu_acceleration_edge_robust_kernel_size;
  std::string floor_edge_robust_kernel;
  double floor_edge_robust_kernel_size;
  std::string loop_closure_edge_robust_kernel;
  double loop_closure_edge_robust_kernel_size;
  bool fix_first_node_adaptive;
  int g2o_solver_num_iterations;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  int max_keyframes_per_update;
  std::deque<KeyFrame::Ptr> new_keyframes;

  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  g2o::VertexPlane* floor_plane_node;
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<rclcpp::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<NmeaSentenceParser> nmea_parser;

  std::unique_ptr<InformationMatrixCalculator> inf_calclator;
};
}  // namespace hdl_graph_slam

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hdl_graph_slam::HdlGraphSlamNodelet>();
  node->onInit();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
