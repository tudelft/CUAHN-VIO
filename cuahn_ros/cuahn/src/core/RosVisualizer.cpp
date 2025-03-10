/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2021 Patrick Geneva
 * Copyright (C) 2021 Guoquan Huang
 * Copyright (C) 2021 OpenVINS Contributors
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2021 Yingfu Xu
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

#include "RosVisualizer.h"

using namespace cuahn;

RosVisualizer::RosVisualizer(ros::NodeHandle &nh, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim) : _app(app), _sim(sim) {

  // Setup our transform broadcaster
  mTfBr = new tf::TransformBroadcaster();

  // Setup pose and path publisher
  pub_poseimu = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/cuahn/poseimu", 2);
  ROS_INFO("Publishing: %s", pub_poseimu.getTopic().c_str());
  pub_odomimu = nh.advertise<nav_msgs::Odometry>("/cuahn/odomimu", 2);
  ROS_INFO("Publishing: %s", pub_odomimu.getTopic().c_str());
  pub_pathimu = nh.advertise<nav_msgs::Path>("/cuahn/pathimu", 2);
  ROS_INFO("Publishing: %s", pub_pathimu.getTopic().c_str());

  // Our tracking image
  pub_tracks = nh.advertise<sensor_msgs::Image>("/cuahn/trackhist", 2);
  ROS_INFO("Publishing: %s", pub_tracks.getTopic().c_str());

  // Groundtruth publishers
  pub_posegt = nh.advertise<geometry_msgs::PoseStamped>("/cuahn/posegt", 2);
  ROS_INFO("Publishing: %s", pub_posegt.getTopic().c_str());
  pub_pathgt = nh.advertise<nav_msgs::Path>("/cuahn/pathgt", 2);
  ROS_INFO("Publishing: %s", pub_pathgt.getTopic().c_str());

  // option to enable publishing of global to IMU transformation
  nh.param<bool>("publish_global_to_imu_tf", publish_global2imu_tf, true);
  nh.param<bool>("publish_calibration_tf", publish_calibration_tf, true);

  // Load groundtruth if we have it and are not doing simulation
  if (nh.hasParam("path_gt") && _sim == nullptr) {
    std::string path_to_gt;
    nh.param<std::string>("path_gt", path_to_gt, "");
    if(!path_to_gt.empty()) {
      DatasetReader::load_gt_file(path_to_gt, gt_states);
      ROS_INFO("gt file path is: %s", path_to_gt.c_str());
    }
  }

  i0_R_w << 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0 ,0.0, -1.0;
}

void RosVisualizer::visualize() {

  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp)
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  // Start timing
  boost::posix_time::ptime rT0_1, rT0_2;
  rT0_1 = boost::posix_time::microsec_clock::local_time();

  // Return if we have not inited
  if (!_app->initialized())
    return;

  // Save the start time of this dataset
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // Print how much time it took to publish / displaying things
  rT0_2 = boost::posix_time::microsec_clock::local_time();
  double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  // printf(BLUE "[TIME]: %.4f seconds for visualization\n" RESET, time_total);
}


void RosVisualizer::visualize_odometry(double timestamp) {

  // Return if we have already visualized
  if (last_odometry_pub_timestamp == _app->get_state()->_timestamp)
    return;
  last_odometry_pub_timestamp = _app->get_state()->_timestamp;

  // Return if we have not inited
  if (!_app->initialized())
    return;

  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // Our odometry message
  nav_msgs::Odometry odomIinM;
  odomIinM.header.stamp = ros::Time(timestamp_inI);
  odomIinM.header.frame_id = "global";

  double roll, pitch, yaw;
  // b_R_w = b_R_i * i_R_i0 * i0_R_w // w(world) and b(body) are front-right-down coordinate frame
  Eigen::Matrix<double, 3, 3> b_R_w = i0_R_w.transpose() * state->_imu->Rot().transpose() * i0_R_w;
  Rot2Euler(b_R_w, roll, pitch, yaw);

  odomIinM.pose.pose.orientation.x = roll;
  odomIinM.pose.pose.orientation.y = pitch;
  odomIinM.pose.pose.orientation.z = yaw;
  // odomIinM.pose.pose.orientation.w = state->_imu->quat()(0);

  // convert position in IMU frame to world frame
  Eigen::Matrix<double, 3, 1> w_pos = state->_imu->Rot() * state->_imu->pos();
  // minus sign for the rotation between IMU and drone body frame (front-right-down)
  odomIinM.pose.pose.position.x = - w_pos(1, 0);
  odomIinM.pose.pose.position.y = - w_pos(0, 0);
  odomIinM.pose.pose.position.z = - w_pos(2, 0);

  // The TWIST component (angular and linear velocities)
  odomIinM.child_frame_id = "imu";
  // convert velocity in IMU frame to world frame
  Eigen::Matrix<double, 3, 1> w_vel = state->_imu->vel(); // for logging velocity in body frame
  odomIinM.twist.twist.linear.x = - w_vel(1, 0);
  odomIinM.twist.twist.linear.y = - w_vel(0, 0);
  odomIinM.twist.twist.linear.z = - w_vel(2, 0);

  // Finally, publish the resulting odometry message
  pub_odomimu.publish(odomIinM);
}

void RosVisualizer::publish_state() {

  // Get the current state
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // Create pose of IMU (note we use the bag time)
  geometry_msgs::PoseWithCovarianceStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_imu;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.pose.orientation.x = state->_imu->quat()(1); 
  poseIinM.pose.pose.orientation.y = state->_imu->quat()(2);
  poseIinM.pose.pose.orientation.z = state->_imu->quat()(3);
  poseIinM.pose.pose.orientation.w = state->_imu->quat()(0);

  // convert position in IMU frame to world frame
  Eigen::Matrix<double, 3, 1> w_pos = state->_imu->Rot() * state->_imu->pos();
  poseIinM.pose.pose.position.x = w_pos(0, 0);
  poseIinM.pose.pose.position.y = w_pos(1, 0);
  poseIinM.pose.pose.position.z = w_pos(2, 0);

  pub_poseimu.publish(poseIinM);

  //=========================================================
  //=========================================================

  // Append to our pose vector
  geometry_msgs::PoseStamped posetemp;
  posetemp.header = poseIinM.header;
  posetemp.pose = poseIinM.pose.pose;
  poses_imu.push_back(posetemp);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::Path arrIMU;
  arrIMU.header.stamp = ros::Time::now();
  arrIMU.header.seq = poses_seq_imu;
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_imu.size(); i += std::floor(poses_imu.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_imu.at(i));
  }
  pub_pathimu.publish(arrIMU);

  // Move them forward in time
  poses_seq_imu++;

  // Publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
  // CUAHN uses imu to world Hamilton quaternion in wxyz sequence
  tf::StampedTransform trans;
  trans.stamp_ = ros::Time::now();
  trans.frame_id_ = "global";
  trans.child_frame_id_ = "imu";
  tf::Quaternion quat(state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->quat()(0));
  trans.setRotation(quat);
  tf::Vector3 orig(w_pos(0), w_pos(1), w_pos(2));
  trans.setOrigin(orig);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }
}

void RosVisualizer::publish_images() {

  // Check if we have subscribers
  if (pub_tracks.getNumSubscribers() == 0)
    return;

  // Get our image of history tracks
  cv::Mat img_history = _app->get_historical_viz_image();

  // Create our message
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

  // Publish
  pub_tracks.publish(exl_msg);
}

void RosVisualizer::publish_groundtruth() {

  // Our groundtruth state
  Eigen::Matrix<double, 17, 1> state_gt;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = _app->get_state()->_timestamp + t_ItoC;

  // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
  if (_sim == nullptr && (gt_states.empty() || !DatasetReader::get_gt_state(timestamp_inI, state_gt, gt_states))) {
    std::cout << "what?!" << std::endl;
    return;
  }

  // Get the GT and system state state
  Eigen::Matrix<double, 16, 1> state_ekf = _app->get_state()->_imu->value();

  // Create pose of IMU
  geometry_msgs::PoseStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_gt;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.orientation.x = state_gt(1, 0);
  poseIinM.pose.orientation.y = state_gt(2, 0);
  poseIinM.pose.orientation.z = state_gt(3, 0);
  poseIinM.pose.orientation.w = state_gt(4, 0);
  poseIinM.pose.position.x = state_gt(5, 0);
  poseIinM.pose.position.y = state_gt(6, 0);
  poseIinM.pose.position.z = state_gt(7, 0);
  pub_posegt.publish(poseIinM);

  // Append to our pose vector
  poses_gt.push_back(poseIinM);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::Path arrIMU;
  arrIMU.header.stamp = ros::Time::now();
  arrIMU.header.seq = poses_seq_gt;
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_gt.size(); i += std::floor(poses_gt.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_gt.at(i));
  }
  pub_pathgt.publish(arrIMU);

  // Move them forward in time
  poses_seq_gt++;

  // Publish our transform on TF
  tf::StampedTransform trans;
  trans.stamp_ = ros::Time::now();
  trans.frame_id_ = "global";
  trans.child_frame_id_ = "truth";
  tf::Quaternion quat(state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0));
  trans.setRotation(quat);
  tf::Vector3 orig(state_gt(5, 0), state_gt(6, 0), state_gt(7, 0));
  trans.setOrigin(orig);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }
}


void RosVisualizer::Rot2Euler(Eigen::Matrix<double, 3, 3> &RotMtrx, double &roll, double &pitch, double &yaw) { // for our naive controller and visualization

  double sy = sqrt(RotMtrx(1, 2)*RotMtrx(1, 2) + RotMtrx(2, 2)*RotMtrx(2, 2));
  if (sy < 1e-6) {
    std::cout << "Pitch is close to 90 degrees!" << std::endl;
    yaw = 0.0;
    roll = atan2(-RotMtrx(2, 1), RotMtrx(1, 1));
  } else {
    yaw = atan2(RotMtrx(0, 1), RotMtrx(0, 0));
    roll = atan2(RotMtrx(1, 2), RotMtrx(2, 2));
  }
  pitch = atan2(-RotMtrx(0, 2), sy);
}
