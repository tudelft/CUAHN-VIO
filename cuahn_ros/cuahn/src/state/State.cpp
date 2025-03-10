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

#include "State.h"
// #include <iomanip>

#include <memory>

using namespace ov_core;
using namespace cuahn;

State::State(StateOptions &options) { // state sequence: [p, q, v, ab, gb, f_ul, f_bl, f_br, f_ur]

  // Save our options
  _options = options;

  // Append the imu to the state and covariance
  int current_id = 0;
  _imu = std::make_shared<IMU>();
  _imu->set_local_id(current_id);
  _variables.push_back(_imu);
  current_id += _imu->size();

  // offsets vectors of (planar homography parameterization)
  _offset_upperLeft = std::make_shared<Vec>(3);
  _variables.push_back(_offset_upperLeft);
  current_id += _offset_upperLeft->size();

  _offset_bottomLeft = std::make_shared<Vec>(3);
  _variables.push_back(_offset_bottomLeft);
  current_id += _offset_bottomLeft->size();

  _offset_bottomRight = std::make_shared<Vec>(3);
  _variables.push_back(_offset_bottomRight);
  current_id += _offset_bottomRight->size();
  
  _offset_upperRight = std::make_shared<Vec>(3);
  _variables.push_back(_offset_upperRight);
  current_id += _offset_upperRight->size();

  // Camera to IMU time offset
  _calib_dt_CAMtoIMU = std::make_shared<Vec>(1);

  // Loop through each camera and create extrinsic and intrinsics 
  // CUAHN uses only one for now
  for (int i = 0; i < _options.num_cameras; i++) {

    // Allocate extrinsic transform
    auto pose = std::make_shared<PoseCUAHN>();

    // Allocate intrinsics for this camera
    auto intrin = std::make_shared<Vec>(8);

    // Add these to the corresponding maps
    _calib_IMUtoCAM.insert({i, pose});
    _cam_intrinsics.insert({i, intrin});
  }

  // Finally initialize our covariance to small value
  _Cov = Eigen::MatrixXd::Zero(27, 27);

  // Get the locations of each entry of the state
  p_id = _imu->p()->id();
  q_id = _imu->q()->id();
  v_id = _imu->v()->id();
  ba_id = _imu->ba()->id();
  bg_id = _imu->bg()->id();
  pt_ul_id = bg_id + 3;
  pt_bl_id = pt_ul_id + 3;
  pt_br_id = pt_bl_id + 3;
  pt_ur_id = pt_br_id + 3;
}

void State::set_extrinsic(const Eigen::Matrix4d &camera_extrinsics) {

  c_RotMtrx_i = camera_extrinsics.block(0, 0, 3, 3);
  i_tVec_i2c = - c_RotMtrx_i.transpose() * camera_extrinsics.block(0, 3, 3, 1);

  std::cout << c_RotMtrx_i << std::endl << i_tVec_i2c << std::endl;
}

void State::reset_4pt_offset() {

  _offset_upperLeft->set_value(Eigen::Vector3d::Zero());
  _offset_bottomLeft->set_value(Eigen::Vector3d::Zero()); 
  _offset_bottomRight->set_value(Eigen::Vector3d::Zero());
  _offset_upperRight->set_value(Eigen::Vector3d::Zero());

  Eigen::MatrixXd new_Cov = Eigen::MatrixXd::Zero(27, 27);
  new_Cov.block(0, 0, 15, 15) = _Cov.block(0, 0, 15, 15);
  _Cov = new_Cov;
}
