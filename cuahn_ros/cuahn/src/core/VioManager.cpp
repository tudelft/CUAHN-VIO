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

#include "VioManager.h"

#include "types/Landmark.h"
#include <memory>

using namespace ov_core;
using namespace ov_type;
using namespace cuahn;

VioManager::VioManager(VioManagerOptions &params_) {

  // Nice startup message
  printf("=======================================\n");
  printf("CUAHN VIO IS STARTING\n");
  printf("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_estimator();
  params.print_noise();
  params.print_state();
  params.print_parameters();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.use_multi_threading ? -1 : 0);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);

  // Loop through through, and load each of the cameras
  for (int i = 0; i < state->_options.num_cameras; i++) {

    // Create the actual camera object and set the values
    if (params.camera_fisheye.at(i)) {
      state->_cam_intrinsics_cameras.insert({i, std::make_shared<CamEqui>()});
      state->_cam_intrinsics_cameras.at(i)->set_value(params.camera_intrinsics.at(i));
      state->_cam_intrinsics_cameras.at(i)->initialize_undist_map_fisheye(); // NOTE when the input images are raw
    } else {
      state->_cam_intrinsics_cameras.insert({i, std::make_shared<CamRadtan>()});
      state->_cam_intrinsics_cameras.at(i)->set_value(params.camera_intrinsics.at(i));
      state->_cam_intrinsics_cameras.at(i)->initialize_undist_map(); // NOTE when the input images are raw
    }

    // Camera intrinsic properties
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i));

    // Our camera extrinsic transform
    state->set_extrinsic(params.camera_extrinsics_SE3);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      printf(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp, loading image, state propagation, network inference, EKF update, total time"<< std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  if(params.use_network) {
    // initialize HNet object
    HNet = std::shared_ptr<pytorch::HomographyNet>(new pytorch::HomographyNet(params.network_model_path, params.network_model_iterative_path, params.use_prior, params.max_IEKF_iteration, params.show_img)); // todo use prior as an input parameter
    printf(BLUE "Network Initialization Finished.\n" RESET);
  } else {
    printf(RED "[ERROR]: Requires use_network in launch file for CUAHN!!!\n" RESET);
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<InertialInitializer>(params.gravity_mag, params.init_window_time, params.init_imu_thresh);

  // Make the updater!
  updaterHNet = std::make_shared<UpdaterHNet>(params.HNet_options);
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // Push back to our propagator
  propagator->feed_imu(message);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message);
  }

  // Count how many unique image streams
  std::vector<int> unique_cam_ids;
  for (const auto &cam_msg : camera_queue) {
    if (std::find(unique_cam_ids.begin(), unique_cam_ids.end(), cam_msg.sensor_ids.at(0)) != unique_cam_ids.end())
      continue;
    unique_cam_ids.push_back(cam_msg.sensor_ids.at(0));
  }

  // If we do not have enough unique cameras then we need to wait
  // We should wait till we have one of each camera to ensure we propagate in the correct order
  size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;
  if (unique_cam_ids.size() != num_unique_cameras)
    return;

  // Loop through our queue and see if we are able to process any of our camera measurements
  // We are able to process if we have at least one IMU measurement greater then the camera time
  double timestamp_inC = message.timestamp - state->_calib_dt_CAMtoIMU->value()(0);
  while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_inC) {
    track_image_and_update(camera_queue.at(0));
    camera_queue.pop_front();
  }
}

// main function
void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // If we do not have VIO initialization, then try to initialize
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize();
    if (!is_initialized_vio)
      return;
  }

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;

  if(params.use_network) { // load image to libtorch tensor , EKF update is in do_feature_propagate_update()
    // message to img Mat
    cv::Mat raw_img = message.images.at(0);
    cv::Mat standard_img;
    if(params.require_undistortion) {
      standard_img = state->_cam_intrinsics_cameras.at(0)->undistort_and_resize_img(raw_img);
    } else {
      standard_img = raw_img;
    }
    HNet->load_current_img(standard_img, message.timestamp); // same function as feed_new_camera
  } else {
    std::cout << "ERROR! CUAHN requires a Neural Network!" << std::endl;
    std::exit(0);
  }

  rT2 = boost::posix_time::microsec_clock::local_time(); // timer: rT2-rT1 load image

  // Call on our propagate and update function
  do_feature_propagate_update(message.timestamp); // not require to pass the whole message from cam, time stamp is enough
}

void VioManager::do_feature_propagate_update(const double time_stamp) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > time_stamp) {
    printf(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET, (time_stamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != time_stamp) {
    propagator->propagate_with_imu(state, time_stamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time(); // timer: rT3-rT2 propagate_with_imu

  double time_nn_inference = 0.0;
  double time_EKF_update = 0.0;
  // iterative EKF
  int num_of_inference = 0;
  Eigen::Matrix<double, 8, 1> propagated_4pt_offset;
  Eigen::Matrix<double, 8, 1> propagated_4pt_offset_pixel;
  while (num_of_inference < params.max_IEKF_iteration)
  {
    // get the prior (propagated 4pt offset)
    propagated_4pt_offset.block(0, 0, 2, 1) = state->_offset_upperLeft->value().block(0, 0, 2, 1);
    propagated_4pt_offset.block(2, 0, 2, 1) = state->_offset_bottomLeft->value().block(0, 0, 2, 1);
    propagated_4pt_offset.block(4, 0, 2, 1) = state->_offset_bottomRight->value().block(0, 0, 2, 1);
    propagated_4pt_offset.block(6, 0, 2, 1) = state->_offset_upperRight->value().block(0, 0, 2, 1);
    propagated_4pt_offset_pixel = propagated_4pt_offset * 159.5; // * 159.5: convert the offset from camera frame to pixels
    // perform HNet network inference here
    HNet->network_inference(propagated_4pt_offset_pixel, num_of_inference); 

    rT4 = boost::posix_time::microsec_clock::local_time(); // timer: rT4-rT3 network_inference

    if (num_of_inference == 0) {
      time_nn_inference = time_nn_inference + (rT4 - rT3).total_microseconds() * 1e-3;
    } else {
      time_nn_inference = time_nn_inference + (rT4 - rT5).total_microseconds() * 1e-3;
    }

    // Return if we where unable to propagate
    if (state->_timestamp != time_stamp) {
      printf(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
      printf(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, time_stamp - state->_timestamp);
      return;
    }

    //===================================================================================
    // update EKF using HNet output
    //===================================================================================

    if (HNet->get_latest_inference_time() == time_stamp && HNet->img_counter > 10) {
      Eigen::Matrix<double, 8, 1> pred_mean = HNet->get_pred_mean();
      Eigen::Matrix<double, 8, 8> pred_Cov = HNet->get_pred_Cov();
      bool update_offset = true;
      if (num_of_inference == params.max_IEKF_iteration - 1) {
        update_offset = false;
      }
      updaterHNet->update(state, pred_mean, pred_Cov, propagated_4pt_offset, update_offset);
      
      rT_EKF_update = boost::posix_time::microsec_clock::local_time(); // timer: rT5-rT4 updaterHNet->update
      time_EKF_update = time_EKF_update + (rT_EKF_update - rT4).total_microseconds() * 1e-3;
    }

    rT5 = boost::posix_time::microsec_clock::local_time();

    num_of_inference = num_of_inference + 1;
  }
  // reset state and Cov after updating
  state->reset_4pt_offset();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_load_img = (rT2 - rT1).total_microseconds() * 1e-3;
  double time_EKF_prop = (rT3 - rT2).total_microseconds() * 1e-3;
  // double time_nn_inference = (rT4 - rT3).total_microseconds() * 1e-3;
  // double time_EKF_update = (rT5 - rT4).total_microseconds() * 1e-3;
  double time_total = (rT5 - rT1).total_microseconds() * 1e-3;

  if (HNet->img_counter > 60) {
      sum_time_load_img = sum_time_load_img + time_load_img;
      sum_time_EKF_prop = sum_time_EKF_prop + time_EKF_prop;
      sum_time_nn_inference = sum_time_nn_inference + time_nn_inference;
      sum_time_EKF_update = sum_time_EKF_update + time_EKF_update;
      sum_time_total = sum_time_total + time_total;

      printf(BLUE "[TIME]: %.3f (%.3f) milliseconds for network inference\n" RESET, time_nn_inference, sum_time_nn_inference/(HNet->img_counter - 60));
      printf(BLUE "[TIME]: %.3f (%.3f) milliseconds for image loading\n" RESET, time_load_img, sum_time_load_img/(HNet->img_counter - 60));
      printf(BLUE "[TIME]: %.3f (%.3f) milliseconds for propagation\n" RESET, time_EKF_prop, sum_time_EKF_prop/(HNet->img_counter - 60));
      printf(BLUE "[TIME]: %.3f (%.3f) milliseconds for EKF update\n" RESET, time_EKF_update, sum_time_EKF_update/(HNet->img_counter - 60));
      printf(BLUE "[TIME]: %.3f (%.3f) milliseconds in total\n" RESET, time_total, sum_time_total/(HNet->img_counter - 60));
      printf("\n" RESET);
  }

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame. The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_load_img << "," 
                  << time_EKF_prop << "," << time_nn_inference << "," << time_EKF_update << "," << time_total << std::endl;
    of_statistics.flush();
  }

  timelastupdate = time_stamp;
}

bool VioManager::try_to_initialize() {

  // Returns from our initializer
  double time0;
  Eigen::Matrix<double, 4, 1> q_I0toW;
  Eigen::Matrix<double, 3, 1> b_g0, I_v_I0, b_a0, I_p_I0;

  // Try to initialize the system
  // We will wait for a jerk if we do not have the zero velocity update enabled
  // Otherwise we can initialize right away as the zero velocity will handle the stationary case
  bool wait_for_jerk = true;
  bool success = initializer->initialize_with_imu_CUAHN(time0, q_I0toW, b_g0, I_v_I0, b_a0, I_p_I0, params.init_height, wait_for_jerk);

  // Return if it failed
  if (!success) {
    return false;
  }

  // Make big vector (q,p,v,bg,ba), and update our state
  // Note: start from zero position, as this is what our covariance is based off of
  Eigen::Matrix<double, 16, 1> imu_val;
  imu_val.block(0, 0, 3, 1) = I_p_I0;
  imu_val.block(3, 0, 4, 1) = q_I0toW;
  imu_val.block(7, 0, 3, 1) = I_v_I0;
  imu_val.block(10, 0, 3, 1) = b_a0;
  imu_val.block(13, 0, 3, 1) = b_g0;

  state->_imu->set_value(imu_val);
  state->_timestamp = time0;
  startup_time = time0;

  // Fix the global yaw and position gauge freedoms
  // set yaw Cov to zero, set x and y position Cov to zeros
  StateHelper::initialize_Cov(state, q_I0toW);

  // Else we are good to go, print out our stats
  printf(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
  printf(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3));
  printf(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
  printf(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));
  printf(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2));
  // printf(GREEN "[INIT]: 4pt offset = %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n" RESET, // 
  //       state->_offset_upperLeft->value()(0), state->_offset_upperLeft->value()(1), state->_offset_bottomLeft->value()(0), state->_offset_bottomLeft->value()(1),
  //       state->_offset_bottomRight->value()(0), state->_offset_bottomRight->value()(1), state->_offset_upperRight->value()(0), state->_offset_upperRight->value()(1));

  propagator->set_const_Jacobian(state);

  return true;
}