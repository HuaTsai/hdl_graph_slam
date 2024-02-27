from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Use simulation time
    use_sim_time = True

    # Arguments
    enable_floor_detection = True
    enable_gps = False
    enable_imu_acc = False
    enable_imu_ori = False
    points_topic = "/aeva/AEVA/point_cloud_compensated"
    imu_topic = "/gpsimu_driver/imu_data"
    map_frame_id = "map"
    lidar_odom_frame_id = "odom"

    # Optional arguments
    enable_robot_odometry_init_guess = False
    robot_odom_frame_id = "robot_odom"

    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lidar2base_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "1", "base_link", points_topic],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("hdl_graph_slam"), "rviz", "hdl_graph_slam.rviz"]
                    ),
                ],
            ),
            # Node(
            #     package="ros2bag",
            #     executable="play",
            #     arguments=["/home/foxconn/Documents/hdl_400", "--delay", "3.0"],
            #     output="screen",
            # ),
            ComposableNodeContainer(
                name="hdl_graph_slam_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container",
                composable_node_descriptions=[
                    ComposableNode(
                        package="hdl_graph_slam",
                        plugin="hdl_graph_slam::PrefilteringNodelet",
                        name="prefiltering_component",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                                "points_topic": points_topic,
                                "imu_topic": imu_topic,
                                "useimu": True,
                                "base_link_frame": "base_link",
                                "use_distance_filter": True,
                                "distance_near_thresh": 0.1,
                                "distance_far_thresh": 100.0,
                                "downsample_method": "VOXELGRID",
                                "downsample_resolution": 0.1,
                                "outlier_removal_method": "RADIUS",
                                "statistical_mean_k": 30,
                                "statistical_stddev": 1.2,
                                "radius_radius": 0.5,
                                "radius_min_neighbors": 2,
                            }
                        ],
                    ),
                    ComposableNode(
                        package="hdl_graph_slam",
                        plugin="hdl_graph_slam::ScanMatchingOdometryNodelet",
                        name="scan_matching_odometry_component",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                                "points_topic": points_topic,
                                "odom_frame_id": lidar_odom_frame_id,
                                "robot_odom_frame_id": robot_odom_frame_id,
                                "keyframe_delta_trans": 1.0,
                                "keyframe_delta_angle": 1.0,
                                "keyframe_delta_time": 10000.0,
                                "transform_thresholding": False,
                                "enable_robot_odometry_init_guess": enable_robot_odometry_init_guess,
                                "max_acceptable_trans": 1.0,
                                "max_acceptable_angle": 1.0,
                                "downsample_method": "NONE",
                                "downsample_resolution": 0.1,
                                "registration_method": "FAST_GICP",
                                "reg_num_threads": 0,
                                "reg_transformation_epsilon": 0.1,
                                "reg_maximum_iterations": 64,
                                "reg_max_correspondence_distance": 2.0,
                                "reg_max_optimizer_iterations": 20,
                                "reg_use_reciprocal_correspondences": False,
                                "reg_correspondence_randomness": 20,
                                "reg_resolution": 1.0,
                                "reg_nn_search_method": "DIRECT7",
                            }
                        ],
                    ),
                    ComposableNode(
                        package="hdl_graph_slam",
                        plugin="hdl_graph_slam::FloorDetectionNodelet",
                        name="floor_detection_component",
                        condition=IfCondition(str(enable_floor_detection)),
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                                "points_topic": points_topic,
                                "tilt_deg": 0.0,
                                "sensor_height": 2.0,
                                "height_clip_range": 1.0,
                                "floor_pts_thresh": 512,
                                "use_normal_filtering": True,
                                "normal_filter_thresh": 20.0,
                            }
                        ],
                    ),
                    # ComposableNode(
                    #     package="hdl_graph_slam",
                    #     plugin="hdl_graph_slam::HdlGraphSlamNodelet",
                    #     name="hdl_graph_slam_component",
                    #     parameters=[
                    #         {
                    #             "use_sim_time": use_sim_time,
                    #             "points_topic": points_topic,
                    #             "map_frame_id": map_frame_id,
                    #             "odom_frame_id": lidar_odom_frame_id,
                    #             "g2o_solver_type": "lm_var_cholmod",
                    #             "g2o_solver_num_iterations": 512,
                    #             "enable_gps": enable_gps,
                    #             "enable_imu_acceleration": enable_imu_acc,
                    #             "enable_imu_orientation": enable_imu_ori,
                    #             "max_keyframes_per_update": 10,
                    #             "keyframe_delta_trans": 2.0,
                    #             "keyframe_delta_angle": 2.0,
                    #             "fix_first_node": True,
                    #             "fix_first_node_stddev": "10 10 10 1 1 1",
                    #             "fix_first_node_adaptive": True,
                    #             "distance_thresh": 15.0,
                    #             "accum_distance_thresh": 25.0,
                    #             "min_edge_interval": 15.0,
                    #             "fitness_score_thresh": 2.5,
                    #             "registration_method": "FAST_GICP",
                    #             "reg_num_threads": 0,
                    #             "reg_transformation_epsilon": 0.1,
                    #             "reg_maximum_iterations": 64,
                    #             "reg_max_correspondence_distance": 2.0,
                    #             "reg_max_optimizer_iterations": 20,
                    #             "reg_use_reciprocal_correspondences": False,
                    #             "reg_correspondence_randomness": 20,
                    #             "reg_resolution": 1.0,
                    #             "reg_nn_search_method": "DIRECT7",
                    #             "gps_edge_robust_kernel": "NONE",
                    #             "gps_edge_robust_kernel_size": 1.0,
                    #             "gps_edge_stddev_xy": 20.0,
                    #             "gps_edge_stddev_z": 5.0,
                    #             "imu_orientation_edge_robust_kernel": "NONE",
                    #             "imu_orientation_edge_stddev": 1.0,
                    #             "imu_acceleration_edge_robust_kernel": "NONE",
                    #             "imu_acceleration_edge_stddev": 1.0,
                    #             "floor_edge_robust_kernel": "NONE",
                    #             "floor_edge_stddev": 10.0,
                    #             "odometry_edge_robust_kernel": "NONE",
                    #             "odometry_edge_robust_kernel_size": 1.0,
                    #             "loop_closure_edge_robust_kernel": "Huber",
                    #             "loop_closure_edge_robust_kernel_size": 1.0,
                    #             "use_const_inf_matrix": False,
                    #             "const_stddev_x": 0.5,
                    #             "const_stddev_q": 0.1,
                    #             "var_gain_a": 20.0,
                    #             "min_stddev_x": 0.1,
                    #             "max_stddev_x": 5.0,
                    #             "min_stddev_q": 0.05,
                    #             "max_stddev_q": 0.2,
                    #             "graph_update_interval": 3.0,
                    #             "map_cloud_update_interval": 10.0,
                    #             "map_cloud_resolution": 0.0,
                    #         }
                    #     ],
                    # ),
                ],
            ),
        ]
    )
