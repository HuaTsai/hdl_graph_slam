// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/keyframe.hpp>

#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace hdl_graph_slam {

KeyFrame::KeyFrame(const rclcpp::Time& stamp, const Eigen::Isometry3d& odom, double accum_distance, const pcl::PointCloud<PointT>::ConstPtr& cloud, rclcpp::Node *parent_node) : stamp(stamp), odom(odom), accum_distance(accum_distance), cloud(cloud), node(nullptr), rosnode(parent_node) {}

KeyFrame::KeyFrame(const std::string& directory, g2o::HyperGraph* graph, rclcpp::Node *parent_node) : stamp(), odom(Eigen::Isometry3d::Identity()), accum_distance(-1), cloud(nullptr), node(nullptr), rosnode(parent_node) {
  load(directory, graph);
}

KeyFrame::~KeyFrame() {}

void KeyFrame::save(const std::string& directory) {
  if(!boost::filesystem::is_directory(directory)) {
    boost::filesystem::create_directory(directory);
  }

  std::ofstream ofs(directory + "/data");
  ofs << "stamp " << stamp.seconds() << " " << stamp.nanoseconds() << "\n";

  ofs << "estimate\n";
  ofs << node->estimate().matrix() << "\n";

  ofs << "odom\n";
  ofs << odom.matrix() << "\n";

  ofs << "accum_distance " << accum_distance << "\n";

  if(floor_coeffs) {
    ofs << "floor_coeffs " << floor_coeffs->transpose() << "\n";
  }

  if(utm_coord) {
    ofs << "utm_coord " << utm_coord->transpose() << "\n";
  }

  if(acceleration) {
    ofs << "acceleration " << acceleration->transpose() << "\n";
  }

  if(orientation) {
    ofs << "orientation " << orientation->w() << " " << orientation->x() << " " << orientation->y() << " " << orientation->z() << "\n";
  }

  if(node) {
    ofs << "id " << node->id() << "\n";
  }

  pcl::io::savePCDFileBinary(directory + "/cloud.pcd", *cloud);
}

bool KeyFrame::load(const std::string& directory, g2o::HyperGraph* graph) {
  std::ifstream ifs(directory + "/data");
  if(!ifs) {
    return false;
  }

  long node_id = -1;
  boost::optional<Eigen::Isometry3d> estimate;

  while(!ifs.eof()) {
    std::string token;
    ifs >> token;

    if(token == "stamp") {
      int sec, nsec;
      ifs >> sec >> nsec;
      stamp = rclcpp::Time(sec, nsec);
    } else if(token == "estimate") {
      Eigen::Matrix4d mat;
      for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
          ifs >> mat(i, j);
        }
      }
      estimate = Eigen::Isometry3d::Identity();
      estimate->linear() = mat.block<3, 3>(0, 0);
      estimate->translation() = mat.block<3, 1>(0, 3);
    } else if(token == "odom") {
      Eigen::Matrix4d odom_mat = Eigen::Matrix4d::Identity();
      for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
          ifs >> odom_mat(i, j);
        }
      }

      odom.setIdentity();
      odom.linear() = odom_mat.block<3, 3>(0, 0);
      odom.translation() = odom_mat.block<3, 1>(0, 3);
    } else if(token == "accum_distance") {
      ifs >> accum_distance;
    } else if(token == "floor_coeffs") {
      Eigen::Vector4d coeffs;
      ifs >> coeffs[0] >> coeffs[1] >> coeffs[2] >> coeffs[3];
      floor_coeffs = coeffs;
    } else if(token == "utm_coord") {
      Eigen::Vector3d coord;
      ifs >> coord[0] >> coord[1] >> coord[2];
      utm_coord = coord;
    } else if(token == "acceleration") {
      Eigen::Vector3d acc;
      ifs >> acc[0] >> acc[1] >> acc[2];
      acceleration = acc;
    } else if(token == "orientation") {
      Eigen::Quaterniond quat;
      ifs >> quat.w() >> quat.x() >> quat.y() >> quat.z();
      orientation = quat;
    } else if(token == "id") {
      ifs >> node_id;
    }
  }

  if(node_id < 0) {
    RCLCPP_ERROR(rosnode->get_logger(), "invalid node id!!");
    RCLCPP_ERROR(rosnode->get_logger(), "%s", directory.c_str());
    return false;
  }

  if(graph->vertices().find(node_id) == graph->vertices().end()) {
    RCLCPP_ERROR(rosnode->get_logger(), "vertex ID = %ld does not exist!!", node_id);
    return false;
  }

  node = dynamic_cast<g2o::VertexSE3*>(graph->vertices()[node_id]);
  if(node == nullptr) {
    RCLCPP_ERROR(rosnode->get_logger(), "failed to downcast!!");
    return false;
  }

  if(estimate) {
    node->setEstimate(*estimate);
  }

  pcl::PointCloud<PointT>::Ptr cloud_(new pcl::PointCloud<PointT>());
  pcl::io::loadPCDFile(directory + "/cloud.pcd", *cloud_);
  cloud = cloud_;

  return true;
}

long KeyFrame::id() const {
  return node->id();
}

Eigen::Isometry3d KeyFrame::estimate() const {
  return node->estimate();
}

KeyFrameSnapshot::KeyFrameSnapshot(const Eigen::Isometry3d& pose, const pcl::PointCloud<PointT>::ConstPtr& cloud) : pose(pose), cloud(cloud) {}

KeyFrameSnapshot::KeyFrameSnapshot(const KeyFrame::Ptr& key) : pose(key->node->estimate()), cloud(key->cloud) {}

KeyFrameSnapshot::~KeyFrameSnapshot() {}

}  // namespace hdl_graph_slam
