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

#include "UpdaterHNet.h"

using namespace ov_core;
using namespace cuahn;

void UpdaterHNet::update(std::shared_ptr<State> state, const Eigen::Matrix<double, 8, 1> &net_4pt_offset_mean,
                         const Eigen::Matrix<double, 8, 8> &net_4pt_offset_Cov, const Eigen::Matrix<double, 8, 1> propagated_4pt_offset, bool update_offset) {

  Eigen::Matrix<double, 27, 8> K_Mtrx = state->_Cov * H.transpose() * (H * state->_Cov * H.transpose() + Hn * (_options.K_net_Cov * net_4pt_offset_Cov / 25440.25) * Hn.transpose()).inverse();

  Eigen::Matrix<double, 8, 1> inno_vec = net_4pt_offset_mean / 159.5 - propagated_4pt_offset; // 8-d

  // update _Cov matrix
  state->_Cov = (Eigen::Matrix<double, 27, 27>::Identity() - K_Mtrx * H) * state->_Cov;

  // updating value  // since we do not need to update 4pt offset (will be set to zero), we can use a small matrix
  Eigen::Matrix<double, 27, 1> d_state_update;
  if (update_offset) {
    d_state_update = K_Mtrx * inno_vec;
  } else {
    d_state_update.block(0, 0, 15, 1) = K_Mtrx.block(0, 0, 15, 8) * inno_vec; 
  }

  Eigen::Matrix<double, 16, 1> updated_imu = state->_imu->value();
  updated_imu.block(0, 0, 3, 1) = state->_imu->pos() + d_state_update.block(0, 0, 3, 1);
  updated_imu.block(3, 0, 4, 1) = quatnorm(Ham_quat_update(d_state_update.block(3, 0, 3, 1)) * state->_imu->quat());
  updated_imu.block(7, 0, 3, 1) = state->_imu->vel() + d_state_update.block(6, 0, 3, 1);
  updated_imu.block(10, 0, 3, 1) = state->_imu->bias_a() + d_state_update.block(9, 0, 3, 1);
  updated_imu.block(13, 0, 3, 1) = state->_imu->bias_g() + d_state_update.block(12, 0, 3, 1);
  
  state->_imu->set_value(updated_imu);

  if (update_offset) {
    state->_offset_upperLeft  ->set_value(state->_offset_upperLeft->value()   + d_state_update.block(15, 0, 3, 1));
    state->_offset_bottomLeft ->set_value(state->_offset_bottomLeft->value()  + d_state_update.block(18, 0, 3, 1));
    state->_offset_bottomRight->set_value(state->_offset_bottomRight->value() + d_state_update.block(21, 0, 3, 1));
    state->_offset_upperRight ->set_value(state->_offset_upperRight->value()  + d_state_update.block(24, 0, 3, 1));
  }
}
