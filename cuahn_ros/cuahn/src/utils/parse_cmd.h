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

#ifndef CUAHN_PARSE_CMDLINE_H
#define CUAHN_PARSE_CMDLINE_H

#include "core/VioManagerOptions.h"
#include "utils/CLI11.hpp"

namespace cuahn {

/**
 * @brief This function will parse the command line arugments using [CLI11](https://github.com/CLIUtils/CLI11).
 * This is only used if you are not building with ROS, and thus isn't the primary supported way to pass arguments.
 * We recommend building with ROS as compared using this parser.
 * @param argc Number of parameters
 * @param argv Pointer to string passed as options
 * @return A fully loaded VioManagerOptions object
 */
VioManagerOptions parse_command_line_arguments(int argc, char **argv) {

  // Our vio manager options with defaults
  VioManagerOptions params;

  // Create our command line parser
  CLI::App app1{"parser_cmd_01"};
  app1.allow_extras();

  // ESTIMATOR ======================================================================

  // Main EKF parameters
  app1.add_option("--use_imuavg", params.state_options.imu_avg, "");
  app1.add_option("--max_cameras", params.state_options.num_cameras, "");

  // Filter initialization
  app1.add_option("--init_window_time", params.init_window_time, "");
  app1.add_option("--init_imu_thresh", params.init_imu_thresh, "");
  app1.add_option("--init_height", params.init_height, "");

  // Recording of timing information to file
  app1.add_option("--record_timing_information", params.record_timing_information, "");
  app1.add_option("--record_timing_filepath", params.record_timing_filepath, "");

  // NOISE ======================================================================

  // Our noise values for inertial sensor
  app1.add_option("--gyroscope_noise_density", params.imu_noises.sigma_w, "");
  app1.add_option("--accelerometer_noise_density", params.imu_noises.sigma_a, "");
  app1.add_option("--gyroscope_random_walk", params.imu_noises.sigma_wb, "");
  app1.add_option("--accelerometer_random_walk", params.imu_noises.sigma_ab, "");

  // Read in update parameters
  app1.add_option("--up_linear_K_HNet_Cov", params.HNet_options.K_net_Cov, "");
  app1.add_option("--max_IEKF_iteration", params.max_IEKF_iteration, "");

  // STATE ======================================================================

  // Timeoffset from camera to IMU
  app1.add_option("--calib_camimu_dt", params.calib_camimu_dt, "");

  // Global gravity
  app1.add_option("--gravity_mag", params.gravity_mag, "");

  // TRACKERS ======================================================================

  // Tracking flags
  app1.add_option("--use_stereo", params.use_stereo, "");
  app1.add_option("--use_network", params.use_network, "");
  app1.add_option("--use_prior", params.use_prior, "");
  app1.add_option("--show_img", params.show_img, "");
  app1.add_option("--require_undistortion", params.require_undistortion, ""); 
  app1.add_option("--network_model_path", params.network_model_path, ""); 
  app1.add_option("--network_model_iterative_path", params.network_model_iterative_path, "");
  app1.add_option("--downsample_cameras", params.downsample_cameras, "");
  app1.add_option("--multi_threading", params.use_multi_threading, "");

  // Preprocessing histogram method
  std::string histogram_method_str = "HISTOGRAM";
  app1.add_option("--histogram_method", histogram_method_str, "");

  // SIMULATION ======================================================================

  // Load the groundtruth trajectory and its spline
  app1.add_option("--sim_traj_path", params.sim_traj_path, "");
  app1.add_option("--sim_distance_threshold", params.sim_distance_threshold, "");
  app1.add_option("--sim_do_perturbation", params.sim_do_perturbation, "");

  // Read in sensor simulation frequencies
  app1.add_option("--sim_freq_cam", params.sim_freq_cam, "");
  app1.add_option("--sim_freq_imu", params.sim_freq_imu, "");

  // Load the seeds for the random number generators
  app1.add_option("--sim_seed_state_init", params.sim_seed_state_init, "");
  app1.add_option("--sim_seed_preturb", params.sim_seed_preturb, "");
  app1.add_option("--sim_seed_measurements", params.sim_seed_measurements, "");

  // CMD PARSE ==============================================================================

  // Finally actually parse the command line and load it
  try {
    app1.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app1.exit(e));
  }

  // Enforce that we have enough cameras to run
  if (params.state_options.num_cameras < 1) {
    printf(RED "VioManager(): Specified number of cameras needs to be greater than zero\n" RESET);
    printf(RED "VioManager(): num cameras = %d\n" RESET, params.state_options.num_cameras);
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing histogram method
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

  //====================================================================================
  //====================================================================================
  //====================================================================================

  // Create our command line parser for the cameras
  // NOTE: we need to first parse how many cameras we have before we can parse this
  CLI::App app2{"parser_cmd_02"};
  app2.allow_extras();

  // Set the defaults
  std::vector<int> p_fish;
  std::vector<std::vector<double>> p_intrinsic;
  std::vector<std::vector<double>> p_extrinsic;
  std::vector<std::vector<int>> p_wh;
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    p_fish.push_back(false);
    p_intrinsic.push_back({458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05});
    p_extrinsic.push_back({0, 0, 0, 1, 0, 0, 0});
    p_wh.push_back({752, 480});
    app2.add_option("--cam" + std::to_string(i) + "_fisheye", p_fish.at(i));
    app2.add_option("--cam" + std::to_string(i) + "_intrinsic", p_intrinsic.at(i), "");
    app2.add_option("--cam" + std::to_string(i) + "_extrinsic", p_extrinsic.at(i), "");
    app2.add_option("--cam" + std::to_string(i) + "_wh", p_wh.at(i), "");
  }

  // Finally actually parse the command line and load it
  try {
    app2.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app2.exit(e));
  }

  // Finally load it into our params
  for (int i = 0; i < params.state_options.num_cameras; i++) {

    // Halve if we are doing downsampling
    p_wh.at(i).at(0) /= (params.downsample_cameras) ? 2.0 : 1.0;
    p_wh.at(i).at(1) /= (params.downsample_cameras) ? 2.0 : 1.0;
    p_intrinsic.at(i).at(0) /= (params.downsample_cameras) ? 2.0 : 1.0;
    p_intrinsic.at(i).at(1) /= (params.downsample_cameras) ? 2.0 : 1.0;
    p_intrinsic.at(i).at(2) /= (params.downsample_cameras) ? 2.0 : 1.0;
    p_intrinsic.at(i).at(3) /= (params.downsample_cameras) ? 2.0 : 1.0;

    // Convert to Eigen
    assert(p_intrinsic.at(i).size() == 8);
    Eigen::Matrix<double, 8, 1> intrinsics;
    intrinsics << p_intrinsic.at(i).at(0), p_intrinsic.at(i).at(1), p_intrinsic.at(i).at(2), p_intrinsic.at(i).at(3),
        p_intrinsic.at(i).at(4), p_intrinsic.at(i).at(5), p_intrinsic.at(i).at(6), p_intrinsic.at(i).at(7);
    assert(p_extrinsic.at(i).size() == 7);
    Eigen::Matrix<double, 7, 1> extrinsics;
    extrinsics << p_extrinsic.at(i).at(0), p_extrinsic.at(i).at(1), p_extrinsic.at(i).at(2), p_extrinsic.at(i).at(3),
        p_extrinsic.at(i).at(4), p_extrinsic.at(i).at(5), p_extrinsic.at(i).at(6);
    assert(p_wh.at(i).size() == 2);

    // Insert
    params.camera_fisheye.insert({i, p_fish.at(i)});
    params.camera_intrinsics.insert({i, intrinsics});
    params.camera_extrinsics.insert({i, extrinsics});
    params.camera_wh.insert({i, {p_wh.at(i).at(0), p_wh.at(i).at(1)}});
  }

  // Success, lets returned the parsed options
  return params;
}

} // namespace cuahn

#endif // CUAHN_PARSE_CMDLINE_H