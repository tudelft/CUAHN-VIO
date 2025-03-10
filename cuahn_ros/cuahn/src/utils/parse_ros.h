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

#if ROS_AVAILABLE || defined(DOXYGEN)
#ifndef CUAHN_PARSE_ROSHANDLER_H
#define CUAHN_PARSE_ROSHANDLER_H

#include <ros/ros.h>

#include "core/VioManagerOptions.h"

namespace cuahn {

/**
 * @brief This function will load paramters from the ros node handler / paramter server
 * This is the recommended way of loading parameters as compared to the command line version.
 * @param nh ROS node handler
 * @return A fully loaded VioManagerOptions object
 */
VioManagerOptions parse_ros_nodehandler(ros::NodeHandle &nh) {

  // Our vio manager options with defaults
  VioManagerOptions params;

  // ESTIMATOR ======================================================================

  // Main EKF parameters
  nh.param<bool>("use_imuavg", params.state_options.imu_avg, params.state_options.imu_avg);
  nh.param<int>("max_cameras", params.state_options.num_cameras, params.state_options.num_cameras);

  // Enforce that we have enough cameras to run
  if (params.state_options.num_cameras < 1) {
    printf(RED "VioManager(): Specified number of cameras needs to be greater than zero\n" RESET);
    printf(RED "VioManager(): num cameras = %d\n" RESET, params.state_options.num_cameras);
    std::exit(EXIT_FAILURE);
  }

  // Filter initialization
  nh.param<double>("init_window_time", params.init_window_time, params.init_window_time);
  nh.param<double>("init_imu_thresh", params.init_imu_thresh, params.init_imu_thresh);
  nh.param<double>("init_height", params.init_height, params.init_height);

  // Recording of timing information to file
  nh.param<bool>("record_timing_information", params.record_timing_information, params.record_timing_information);
  nh.param<std::string>("record_timing_filepath", params.record_timing_filepath, params.record_timing_filepath);

  // NOISE ======================================================================

  // Our noise values for inertial sensor
  nh.param<double>("gyroscope_noise_density", params.imu_noises.sigma_w, params.imu_noises.sigma_w);
  nh.param<double>("accelerometer_noise_density", params.imu_noises.sigma_a, params.imu_noises.sigma_a);
  nh.param<double>("gyroscope_random_walk", params.imu_noises.sigma_wb, params.imu_noises.sigma_wb);
  nh.param<double>("accelerometer_random_walk", params.imu_noises.sigma_ab, params.imu_noises.sigma_ab);

  // Read in update parameters
  nh.param<double>("up_linear_K_HNet_Cov", params.HNet_options.K_net_Cov, params.HNet_options.K_net_Cov);
  nh.param<int>("max_IEKF_iteration", params.max_IEKF_iteration, params.max_IEKF_iteration);

  // STATE ======================================================================

  // Timeoffset from camera to IMU
  nh.param<double>("calib_camimu_dt", params.calib_camimu_dt, params.calib_camimu_dt);

  // Global gravity
  nh.param<double>("gravity_mag", params.gravity_mag, params.gravity_mag);

  // TRACKERS ======================================================================

  // Tracking flags
  nh.param<bool>("use_stereo", params.use_stereo, params.use_stereo);
  nh.param<std::string>("network_model_path", params.network_model_path, params.network_model_path);
  nh.param<std::string>("network_model_iterative_path", params.network_model_iterative_path, params.network_model_iterative_path);
  nh.param<bool>("use_network", params.use_network, params.use_network); 
  nh.param<bool>("use_prior", params.use_prior, params.use_prior); 
  nh.param<bool>("show_img", params.show_img, params.show_img); 
  nh.param<bool>("require_undistortion", params.require_undistortion, params.require_undistortion);
  nh.param<bool>("downsample_cameras", params.downsample_cameras, params.downsample_cameras);
  nh.param<bool>("multi_threading", params.use_multi_threading, params.use_multi_threading);

  // General parameters
  // nh.param<int>("num_pts", params.num_pts, params.num_pts);
  // nh.param<int>("fast_threshold", params.fast_threshold, params.fast_threshold);
  // nh.param<int>("grid_x", params.grid_x, params.grid_x);
  // nh.param<int>("grid_y", params.grid_y, params.grid_y);
  // nh.param<int>("min_px_dist", params.min_px_dist, params.min_px_dist);
  // nh.param<double>("knn_ratio", params.knn_ratio, params.knn_ratio);

  // Preprocessing histogram method
  std::string histogram_method_str = "HISTOGRAM";
  nh.param<std::string>("histogram_method", histogram_method_str, histogram_method_str);
  if (histogram_method_str == "NONE") {
    params.histogram_method = TrackBase::NONE;
  } else if (histogram_method_str == "HISTOGRAM") {
    params.histogram_method = TrackBase::HISTOGRAM;
  } else if (histogram_method_str == "CLAHE") {
    params.histogram_method = TrackBase::CLAHE;
  } else {
    printf(RED "VioManager(): invalid feature histogram specified:\n" RESET);
    printf(RED "\t- NONE\n" RESET);
    printf(RED "\t- HISTOGRAM\n" RESET);
    printf(RED "\t- CLAHE\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Feature mask
  nh.param<bool>("use_mask", params.use_mask, params.use_mask);
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string mask_path;
    nh.param<std::string>("mask" + std::to_string(i), mask_path, "");
    if (params.use_mask) {
      if (!boost::filesystem::exists(mask_path)) {
        printf(RED "VioManager(): invalid mask path:\n" RESET);
        printf(RED "\t- mask%d\n" RESET, i);
        printf(RED "\t- %s\n" RESET, mask_path.c_str());
        std::exit(EXIT_FAILURE);
      }
      params.masks.insert({i, cv::imread(mask_path, cv::IMREAD_GRAYSCALE)});
    }
  }

  // Feature initializer parameters
  nh.param<bool>("fi_triangulate_1d", params.featinit_options.triangulate_1d, params.featinit_options.triangulate_1d);
  nh.param<bool>("fi_refine_features", params.featinit_options.refine_features, params.featinit_options.refine_features);
  nh.param<int>("fi_max_runs", params.featinit_options.max_runs, params.featinit_options.max_runs);
  nh.param<double>("fi_init_lamda", params.featinit_options.init_lamda, params.featinit_options.init_lamda);
  nh.param<double>("fi_max_lamda", params.featinit_options.max_lamda, params.featinit_options.max_lamda);
  nh.param<double>("fi_min_dx", params.featinit_options.min_dx, params.featinit_options.min_dx);
  nh.param<double>("fi_min_dcost", params.featinit_options.min_dcost, params.featinit_options.min_dcost);
  nh.param<double>("fi_lam_mult", params.featinit_options.lam_mult, params.featinit_options.lam_mult);
  nh.param<double>("fi_min_dist", params.featinit_options.min_dist, params.featinit_options.min_dist);
  nh.param<double>("fi_max_dist", params.featinit_options.max_dist, params.featinit_options.max_dist);
  nh.param<double>("fi_max_baseline", params.featinit_options.max_baseline, params.featinit_options.max_baseline);
  nh.param<double>("fi_max_cond_number", params.featinit_options.max_cond_number, params.featinit_options.max_cond_number);

  // SIMULATION ======================================================================

  // Load the groundtruth trajectory and its spline
  nh.param<std::string>("sim_traj_path", params.sim_traj_path, params.sim_traj_path);
  nh.param<double>("sim_distance_threshold", params.sim_distance_threshold, params.sim_distance_threshold);
  nh.param<bool>("sim_do_perturbation", params.sim_do_perturbation, params.sim_do_perturbation);

  // Read in sensor simulation frequencies
  nh.param<double>("sim_freq_cam", params.sim_freq_cam, params.sim_freq_cam);
  nh.param<double>("sim_freq_imu", params.sim_freq_imu, params.sim_freq_imu);

  // Load the seeds for the random number generators
  nh.param<int>("sim_seed_state_init", params.sim_seed_state_init, params.sim_seed_state_init);
  nh.param<int>("sim_seed_preturb", params.sim_seed_preturb, params.sim_seed_preturb);
  nh.param<int>("sim_seed_measurements", params.sim_seed_measurements, params.sim_seed_measurements);

  //====================================================================================
  //====================================================================================
  //====================================================================================

  // Loop through through, and load each of the cameras
  for (int i = 0; i < params.state_options.num_cameras; i++) {

    // If our distortions are fisheye or not!
    bool is_fisheye;
    nh.param<bool>("cam" + std::to_string(i) + "_is_fisheye", is_fisheye, false);

    // If the desired fov we should simulate
    std::vector<int> matrix_wh;
    std::vector<int> matrix_wd_default = {752, 480};
    nh.param<std::vector<int>>("cam" + std::to_string(i) + "_wh", matrix_wh, matrix_wd_default);
    matrix_wh.at(0) /= (params.downsample_cameras) ? 2.0 : 1.0;
    matrix_wh.at(1) /= (params.downsample_cameras) ? 2.0 : 1.0;
    std::pair<int, int> wh(matrix_wh.at(0), matrix_wh.at(1));

    // Camera intrinsic properties
    Eigen::Matrix<double, 8, 1> cam_calib;
    std::vector<double> matrix_k, matrix_d;
    std::vector<double> matrix_k_default = {458.654, 457.296, 367.215, 248.375};
    std::vector<double> matrix_d_default = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    nh.param<std::vector<double>>("cam" + std::to_string(i) + "_k", matrix_k, matrix_k_default);
    nh.param<std::vector<double>>("cam" + std::to_string(i) + "_d", matrix_d, matrix_d_default);
    matrix_k.at(0) /= (params.downsample_cameras) ? 2.0 : 1.0;
    matrix_k.at(1) /= (params.downsample_cameras) ? 2.0 : 1.0;
    matrix_k.at(2) /= (params.downsample_cameras) ? 2.0 : 1.0;
    matrix_k.at(3) /= (params.downsample_cameras) ? 2.0 : 1.0;
    cam_calib << matrix_k.at(0), matrix_k.at(1), matrix_k.at(2), matrix_k.at(3), matrix_d.at(0), matrix_d.at(1), matrix_d.at(2),
        matrix_d.at(3);

    // Our camera extrinsics transform
    Eigen::Matrix4d T_CtoI;
    std::vector<double> matrix_TCtoI;
    std::vector<double> matrix_TtoI_default = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    // Read in from ROS, and save into our eigen mat
    nh.param<std::vector<double>>("T_C" + std::to_string(i) + "toI", matrix_TCtoI, matrix_TtoI_default);
    T_CtoI << matrix_TCtoI.at(0), matrix_TCtoI.at(1), matrix_TCtoI.at(2), matrix_TCtoI.at(3), matrix_TCtoI.at(4), matrix_TCtoI.at(5),
        matrix_TCtoI.at(6), matrix_TCtoI.at(7), matrix_TCtoI.at(8), matrix_TCtoI.at(9), matrix_TCtoI.at(10), matrix_TCtoI.at(11),
        matrix_TCtoI.at(12), matrix_TCtoI.at(13), matrix_TCtoI.at(14), matrix_TCtoI.at(15);

    // Load these into our state
    Eigen::Matrix<double, 7, 1> cam_eigen;
    cam_eigen.block(0, 0, 4, 1) = rot_2_quat(T_CtoI.block(0, 0, 3, 3).transpose());
    cam_eigen.block(4, 0, 3, 1) = -T_CtoI.block(0, 0, 3, 3).transpose() * T_CtoI.block(0, 3, 3, 1);

    // Insert
    params.camera_fisheye.insert({i, is_fisheye});
    params.camera_intrinsics.insert({i, cam_calib});
    params.camera_extrinsics.insert({i, cam_eigen});
    
    params.camera_wh.insert({i, wh});
  }

  // Our camera extrinsics transform
  Eigen::Matrix4d T_ItoC;
  std::vector<double> matrix_TItoC;
  std::vector<double> matrix_TItoC_default = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // Read in from ROS, and save into our eigen mat
  nh.param<std::vector<double>>("T_ItoCmono", matrix_TItoC, matrix_TItoC_default);
  T_ItoC << matrix_TItoC.at(0), matrix_TItoC.at(1), matrix_TItoC.at(2), matrix_TItoC.at(3), matrix_TItoC.at(4), matrix_TItoC.at(5),
      matrix_TItoC.at(6), matrix_TItoC.at(7), matrix_TItoC.at(8), matrix_TItoC.at(9), matrix_TItoC.at(10), matrix_TItoC.at(11),
      matrix_TItoC.at(12), matrix_TItoC.at(13), matrix_TItoC.at(14), matrix_TItoC.at(15);
  params.camera_extrinsics_SE3 = T_ItoC;// CUAHN simple 

  // Success, lets returned the parsed options
  return params;
}

} // namespace cuahn

#endif // CUAHN_PARSE_ROSHANDLER_H
#endif // ROS_AVAILABLE