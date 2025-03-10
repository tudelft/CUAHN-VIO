/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2021 Patrick Geneva
 * Copyright (C) 2021 Guoquan Huang
 * Copyright (C) 2021 OpenVINS Contributors
 * Copyright (C) 2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUAHN_ROSVISUALIZER_H
#define CUAHN_ROSVISUALIZER_H

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>

#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

#include "VioManager.h"
#include "sim/Simulator.h"
#include "utils/dataset_reader.h"

#include <math.h> // for quat 2 Euler

namespace cuahn {

/**
 * @brief Helper class that will publish results onto the ROS framework.
 *
 * Also save to file the current total state and covariance along with the groundtruth if we are simulating.
 * We visualize the following things:
 * - State of the system on TF, pose message, and path
 * - Image of our tracker
 * - Our different features (SLAM, MSCKF, ARUCO)
 * - Groundtruth trajectory if we have it
 */
class RosVisualizer {

public:
  /**
   * @brief Default constructor
   * @param nh ROS node handler
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  RosVisualizer(ros::NodeHandle &nh, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);

  /**
   * @brief Will visualize the system if we have new things
   */
  void visualize();

  /**
   * @brief Will publish our odometry message for the current timestep.
   * This will take the current state estimate and get the propagated pose to the desired time.
   * This can be used to get pose estimates on systems which require high frequency pose estimates.
   */
  void visualize_odometry(double timestamp);

protected:
  /// Publish the current state
  void publish_state();

  /// Publish the active tracking image
  void publish_images();

  /// Publish current features
  void publish_features();

  /// Publish groundtruth (if we have it)
  void publish_groundtruth();

  void Rot2Euler(Eigen::Matrix<double, 3, 3> &RotMtrx, double &roll, double &pitch, double &yaw);

  Eigen::Matrix<double, 3, 3> i0_R_w;

  /// Core application of the filter system
  std::shared_ptr<VioManager> _app;

  /// Simulator (is nullptr if we are not sim'ing)
  std::shared_ptr<Simulator> _sim;

  // Our publishers
  ros::Publisher pub_poseimu, pub_odomimu, pub_pathimu;
  ros::Publisher pub_points_msckf, pub_points_slam, pub_points_aruco, pub_points_sim;
  ros::Publisher pub_tracks;
  ros::Publisher pub_loop_pose, pub_loop_point, pub_loop_extrinsic, pub_loop_intrinsics;
  ros::Publisher pub_loop_img_depth, pub_loop_img_depth_color;
  tf::TransformBroadcaster *mTfBr;

  // For path viz
  unsigned int poses_seq_imu = 0;
  vector<geometry_msgs::PoseStamped> poses_imu;

  // Groundtruth infomation
  ros::Publisher pub_pathgt, pub_posegt;
  double summed_rmse_ori = 0.0;
  double summed_rmse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // Start and end timestamps
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // Last timestamp we visualized at
  double last_visualization_timestamp = 0;
  double last_odometry_pub_timestamp = 0;

  // Our groundtruth states
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // For path viz
  unsigned int poses_seq_gt = 0;
  vector<geometry_msgs::PoseStamped> poses_gt;
  bool publish_global2imu_tf = true;
  bool publish_calibration_tf = true;

};

} // namespace cuahn

#endif // CUAHN_ROSVISUALIZER_H
