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

#include "StateHelper.h"

using namespace ov_core;
using namespace cuahn;

void StateHelper::propagate_Cov(std::shared_ptr<State> state, const Eigen::MatrixXd &F, const Eigen::MatrixXd &Fw, const Eigen::MatrixXd &Q) {
  // std::cout << 
  state->_Cov = F * state->_Cov * F.transpose() + Fw * Q * Fw.transpose();
  // self.P_Mtrx = F_Mtrx @ self.P_Mtrx @ F_Mtrx.T + Fw_Mtrx @ self.Q_Mtrx @ Fw_Mtrx.T
}

void StateHelper::initialize_Cov(std::shared_ptr<State> state, const Eigen::Vector4d &q_I0toW) {

  // set p Cov
  state->_Cov.block(state->_imu->p()->id(), state->_imu->p()->id(), 2, 2).setZero(); // only set x and y position Cov to zero
  state->_Cov(state->_imu->p()->id()+2, state->_imu->p()->id()+2) = 0.005 * 0.005; //
 
  // set q Cov
  double std_degree = 0.5;
  state->_Cov(state->_imu->q()->id(), state->_imu->q()->id()) = (std_degree / 180.0 * 3.14159265) * (std_degree / 180.0 * 3.14159265); // same as Python
  state->_Cov(state->_imu->q()->id() + 1, state->_imu->q()->id() + 1) = (std_degree / 180.0 * 3.14159265) * (std_degree / 180.0 * 3.14159265);
  state->_Cov(state->_imu->q()->id() + 2, state->_imu->q()->id() + 2) = 0.0;
  
  // set imu bias Cov
  state->_Cov.block(state->_imu->ba()->id(), state->_imu->ba()->id(), 3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 0.005 * 0.005;
  state->_Cov.block(state->_imu->bg()->id(), state->_imu->bg()->id(), 3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 0.0000 * 0.0000;

  // Propagate into the current local IMU frame
  Eigen::Matrix3d w_R_i = Ham_quat_2_Rot(q_I0toW);

  // Propagate p
  state->_Cov.block(state->_imu->p()->id(), state->_imu->p()->id(), 3, 3) =
      w_R_i.transpose() * state->_Cov.block(state->_imu->p()->id(), state->_imu->p()->id(), 3, 3) * w_R_i;

  // Propagate q
  state->_Cov.block(state->_imu->q()->id(), state->_imu->q()->id(), 3, 3) =
      w_R_i.transpose() * state->_Cov.block(state->_imu->q()->id(), state->_imu->q()->id(), 3, 3) * w_R_i;

  printf(YELLOW "Initialized _Cov.diagonal(): \n", RESET);
  std::cout << state->_Cov.diagonal().transpose() << std::endl;
}