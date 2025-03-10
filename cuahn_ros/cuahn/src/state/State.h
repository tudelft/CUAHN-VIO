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

#ifndef CUAHN_STATE_H
#define CUAHN_STATE_H

#include <map>
#include <unordered_map>
#include <vector>

#include "StateOptions.h"
#include "cam/CamBase.h"
#include "types/IMU_CUAHN.h"
#include "types/Landmark.h"
#include "types/PoseCUAHN.h"
#include "types/Type.h"
#include "types/Vec.h"

using namespace ov_core;
using namespace ov_type;

namespace cuahn {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
class State {

public:
  /**
   * @brief Default Constructor (will initialize variables to defaults)
   * @param options_ Options structure containing filter options
   */
  State(StateOptions &options_);

  ~State() {}

  /**
   * @brief Calculates the current max size of the covariance
   * @return Size of the current covariance matrix
   */
  int max_covariance_size() { return (int)_Cov.rows(); }

  /// Current timestamp (should be the last update time!)
  double _timestamp;

  /// Struct containing filter options
  StateOptions _options;

  /// Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
  std::shared_ptr<IMU> _imu;

  /// 4 corner points pixel location offset vectors (2-d vectors)
  std::shared_ptr<Vec> _offset_upperLeft;
  std::shared_ptr<Vec> _offset_bottomLeft;
  std::shared_ptr<Vec> _offset_bottomRight;
  std::shared_ptr<Vec> _offset_upperRight;

  /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
  std::map<double, std::shared_ptr<PoseCUAHN>> _clones_IMU;

  /// Our current set of SLAM features (3d positions)
  std::unordered_map<size_t, std::shared_ptr<Landmark>> _features_SLAM;

  /// Time offset base IMU to camera (t_imu = t_cam + t_off)
  std::shared_ptr<Vec> _calib_dt_CAMtoIMU;

  /// Calibration poses for each camera (R_ItoC, p_IinC)
  std::unordered_map<size_t, std::shared_ptr<PoseCUAHN>> _calib_IMUtoCAM;

  /// since CUAHN does not update extrinsics, we define them simply as Eigen vecter and matrix
  void set_extrinsic(const Eigen::Matrix4d &camera_extrinsics);

  void reset_4pt_offset();

  /// Camera intrinsics
  std::unordered_map<size_t, std::shared_ptr<Vec>> _cam_intrinsics;

  /// Camera intrinsics camera objects
  std::unordered_map<size_t, std::shared_ptr<CamBase>> _cam_intrinsics_cameras;

  /// since CUAHN does not update extrinsics, we define them simply as Eigen vecter and matrix
  Eigen::Matrix<double, 3, 1> i_tVec_i2c;
  Eigen::Matrix<double, 3, 3> c_RotMtrx_i;

  const Eigen::Matrix<double, 3, 1> cam_upperLeft_pt_xy1   = (Eigen::Matrix<double, 3, 1>() << -1.0, -0.69906, 1.0).finished();
  const Eigen::Matrix<double, 3, 1> cam_bottomLeft_pt_xy1  = (Eigen::Matrix<double, 3, 1>() << -1.0,  0.69906, 1.0).finished();
  const Eigen::Matrix<double, 3, 1> cam_bottomRight_pt_xy1 = (Eigen::Matrix<double, 3, 1>() <<  1.0,  0.69906, 1.0).finished();
  const Eigen::Matrix<double, 3, 1> cam_upperRight_pt_xy1  = (Eigen::Matrix<double, 3, 1>() <<  1.0, -0.69906, 1.0).finished();

  int p_id;
  int q_id;
  int v_id;
  int ba_id;
  int bg_id;
  int pt_ul_id;
  int pt_bl_id;
  int pt_br_id;
  int pt_ur_id;

private:
  // Define that the state helper is a friend class of this class
  // This will allow it to access the below functions which should normally not be called
  // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
  friend class StateHelper;

  friend class UpdaterHNet;

  /// Covariance of all active variables
  Eigen::MatrixXd _Cov;

  /// Vector of variables
  std::vector<std::shared_ptr<Type>> _variables;
};

} // namespace cuahn

#endif // CUAHN_STATE_H