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

#ifndef CUAHN_VIOMANAGER_H
#define CUAHN_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <string>

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "init/InertialInitializer.h"
#include "utils/lambda_body.h"
#include "utils/sensor_data.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHNet.h"
#include "VioManagerOptions.h"
#include "HomographyNet.h"

namespace cuahn {

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for CUAHN to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) {
    camera_queue.push_back(message);
    std::sort(camera_queue.begin(), camera_queue.end());
  }

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */

  /// If we are initialized or not
  bool initialized() { return is_initialized_vio; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  // /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image() {

    // Get our image of history tracks
    cv::Mat img_history;

    // Build an id-list of what features we should highlight (i.e. SLAM)
    std::vector<size_t> highlighted_ids;
    for (const auto &feat : state->_features_SLAM) {
      highlighted_ids.push_back(feat.first);
    }
    // Finally return the image
    return img_history;
  }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const double time_stamp);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @return True if we have successfully initialized
   */
  bool try_to_initialize();

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// State initializer
  std::shared_ptr<InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Our Homography Network updater
  std::shared_ptr<UpdaterHNet> updaterHNet;

  /// Queue up camera measurements sorted by time and trigger once we have
  /// exactly one IMU measurement with timestamp newer than the camera measurement
  /// This also handles out-of-order camera measurements, which is rare, but
  /// a nice feature to have for general robustness to bad camera drivers.
  std::deque<ov_core::CameraData> camera_queue;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7, rT_EKF_update;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  // double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  /// Our HNet for homography predition 
  std::shared_ptr<pytorch::HomographyNet> HNet;

  cv::Mat img1_nn;
  double timeStamp_img1_nn;
  cv::Mat img2_nn;
  double timeStamp_img2_nn;

  double sum_time_load_img = 0;
  double sum_time_EKF_prop = 0;
  double sum_time_nn_inference = 0;
  double sum_time_EKF_update = 0;
  double sum_time_total = 0;

};

} // namespace cuahn

#endif // CUAHN_VIOMANAGER_H
