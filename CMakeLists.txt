# SPDX-License-Identifier: BSD-2-Clause
cmake_minimum_required(VERSION 3.8)
project(hdl_graph_slam)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS "-march=native")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nmea_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geodesy REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(PCL REQUIRED)
find_package(G2O REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  rclcpp_components
  std_srvs
  nmea_msgs
  nav_msgs
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  geodesy
  ndt_omp
  fast_gicp
  PCL
  G2O
)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_library(VGICP_CUDA_FOUND NAMES fast_vgicp_cuda)
message(STATUS "VGICP_CUDA_FOUND:" ${VGICP_CUDA_FOUND})
if(VGICP_CUDA_FOUND)
  add_definitions(-DUSE_VGICP_CUDA)
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FloorCoeffs.msg
  msg/ScanMatchingStatus.msg
  srv/SaveMap.srv
  srv/LoadGraph.srv
  srv/DumpGraph.srv
  DEPENDENCIES std_msgs geometry_msgs
)

if ($ENV{ROS_DISTRO} STREQUAL "humble")
  rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
endif()

include_directories(include)

function(link_typesupport target)
  if ($ENV{ROS_DISTRO} STREQUAL "humble")
    target_link_libraries(${target} ${cpp_typesupport_target})
  else()
    rosidl_target_interfaces(${target} ${PROJECT_NAME} "rosidl_typesupport_cpp")
  endif()
endfunction()

# Prefiltering Component
add_library(prefiltering_nodelet SHARED apps/prefiltering_nodelet.cpp)
ament_target_dependencies(prefiltering_nodelet ${dependencies})
link_typesupport(prefiltering_nodelet)
rclcpp_components_register_nodes(prefiltering_nodelet hdl_graph_slam::PrefilteringNodelet)

# Floor Detection Component
add_library(floor_detection_nodelet SHARED apps/floor_detection_nodelet.cpp)
ament_target_dependencies(floor_detection_nodelet ${dependencies})
link_typesupport(floor_detection_nodelet)
rclcpp_components_register_nodes(floor_detection_nodelet hdl_graph_slam::FloorDetectionNodelet)

# Scan Matching Odometry Component
add_library(scan_matching_odometry_nodelet SHARED
  apps/scan_matching_odometry_nodelet.cpp
  src/hdl_graph_slam/registrations.cpp
)
ament_target_dependencies(scan_matching_odometry_nodelet ${dependencies})
link_typesupport(scan_matching_odometry_nodelet)
rclcpp_components_register_nodes(scan_matching_odometry_nodelet hdl_graph_slam::ScanMatchingOdometryNodelet)

# HDL Graph SLAM Component
add_library(hdl_graph_slam_nodelet SHARED
  apps/hdl_graph_slam_nodelet.cpp
  src/hdl_graph_slam/graph_slam.cpp
  src/hdl_graph_slam/keyframe.cpp
  src/hdl_graph_slam/map_cloud_generator.cpp
  src/hdl_graph_slam/registrations.cpp
  src/hdl_graph_slam/information_matrix_calculator.cpp
  src/g2o/robust_kernel_io.cpp
)
ament_target_dependencies(hdl_graph_slam_nodelet ${dependencies})
link_typesupport(hdl_graph_slam_nodelet)
target_link_libraries(hdl_graph_slam_nodelet
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}   # be aware of that CSPARSE is released under LGPL
  ${G2O_SOLVER_CHOLMOD}   # be aware of that cholmod is released under GPL
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
)
rclcpp_components_register_nodes(hdl_graph_slam_nodelet hdl_graph_slam::HdlGraphSlamNodelet)

install(
  TARGETS
    prefiltering_nodelet
    floor_detection_nodelet
    scan_matching_odometry_nodelet
    hdl_graph_slam_nodelet
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

