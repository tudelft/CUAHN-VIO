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

#include "Propagator.h"

using namespace ov_core;
using namespace cuahn;

void Propagator::propagate_with_imu(std::shared_ptr<State> state, double timestamp) {

  // If the difference between the current update time and state is zero
  // We should crash, as this means we would have two clones at the same time!!!!
  if (state->_timestamp == timestamp) {
    printf(RED "Propagator::propagate_with_imu(): Propagation called again at same timestep at last update timestep!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // We should crash if we are trying to propagate backwards
  if (state->_timestamp > timestamp) {
    printf(RED "Propagator::propagate_with_imu(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
    printf(RED "Propagator::propagate_with_imu(): desired propagation = %.4f\n" RESET, (timestamp - state->_timestamp));
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Set the last time offset value if we have just started the system up
  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  double time0 = state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;
  std::vector<ov_core::ImuData> prop_data = Propagator::select_imu_readings(imu_data, time0, time1);
  
  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  if (prop_data.size() > 1) {
    for (size_t i = 0; i < prop_data.size() - 1; i++) {

      // Get the next state Jacobian and noise Jacobian for this IMU reading
      predict_and_compute(state, prop_data.at(i), prop_data.at(i + 1), F, Fw);
      StateHelper::propagate_Cov(state, F, Fw, Q);
    }
  }

  std::cout << "Speed/Height = " << state->_imu->vel().norm() / dc << std::endl;

  // Set timestamp data
  state->_timestamp = timestamp;
  last_prop_time_offset = t_off_new;

}

std::vector<ov_core::ImuData> Propagator::select_imu_readings(const std::vector<ov_core::ImuData> &imu_data, double time0, double time1,
                                                              bool warn) {

  // Our vector imu readings
  std::vector<ov_core::ImuData> prop_data;

  // Ensure we have some measurements in the first place!
  if (imu_data.empty()) {
    if (warn)
      printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements. IMU-CAMERA are likely messed up!!!\n" RESET);
    return prop_data;
  }

  // Loop through and find all the needed measurements to propagate with
  // Note we split measurements based on the given state time, and the update timestamp
  for (size_t i = 0; i < imu_data.size() - 1; i++) {

    // START OF THE INTEGRATION PERIOD
    // If the next timestamp is greater then our current state time
    // And the current is not greater then it yet...
    // Then we should "split" our current IMU measurement
    if (imu_data.at(i + 1).timestamp > time0 && imu_data.at(i).timestamp < time0) {
      ov_core::ImuData data = Propagator::interpolate_data(imu_data.at(i), imu_data.at(i + 1), time0);
      prop_data.push_back(data);
      continue;
    }

    // MIDDLE OF INTEGRATION PERIOD
    // If our imu measurement is right in the middle of our propagation period
    // Then we should just append the whole measurement time to our propagation vector
    if (imu_data.at(i).timestamp >= time0 && imu_data.at(i + 1).timestamp <= time1) {
      prop_data.push_back(imu_data.at(i));
      continue;
    }

    // END OF THE INTEGRATION PERIOD
    // If the current timestamp is greater then our update time
    // We should just "split" the NEXT IMU measurement to the update time,
    // NOTE: we add the current time, and then the time at the end of the interval (so we can get a dt)
    // NOTE: we also break out of this loop, as this is the last IMU measurement we need!
    if (imu_data.at(i + 1).timestamp > time1) {
      // If we have a very low frequency IMU then, we could have only recorded the first integration (i.e. case 1) and nothing else
      // In this case, both the current IMU measurement and the next is greater than the desired intepolation, thus we should just cut the
      // current at the desired time Else, we have hit CASE2 and this IMU measurement is not past the desired propagation time, thus add the
      // whole IMU reading
      if (imu_data.at(i).timestamp > time1 && i == 0) {
        // This case can happen if we don't have any imu data that has occured before the startup time
        // This means that either we have dropped IMU data, or we have not gotten enough.
        // In this case we can't propgate forward in time, so there is not that much we can do.
        break;
      } else if (imu_data.at(i).timestamp > time1) {
        ov_core::ImuData data = interpolate_data(imu_data.at(i - 1), imu_data.at(i), time1);
        prop_data.push_back(data);
      } else {
        prop_data.push_back(imu_data.at(i));
      }
      // If the added IMU message doesn't end exactly at the camera time
      // Then we need to add another one that is right at the ending time
      if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
        ov_core::ImuData data = interpolate_data(imu_data.at(i), imu_data.at(i + 1), time1);
        prop_data.push_back(data);
      }
      break;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.empty()) {
    if (warn)
      printf(
          YELLOW
          "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // Loop through and ensure we do not have an zero dt values
  // This would cause the noise covariance to be Infinity
  for (size_t i = 0; i < prop_data.size() - 1; i++) {
    if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) < 1e-12) {
      if (warn)
        printf(YELLOW "Propagator::select_imu_readings(): Zero DT between IMU reading %d and %d, removing it!\n" RESET, (int)i,
               (int)(i + 1));
      prop_data.erase(prop_data.begin() + i);
      i--;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.size() < 2) {
    if (warn)
      printf(
          YELLOW
          "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // Success :D
  return prop_data;
}

void Propagator::predict_and_compute(std::shared_ptr<State> state, const ov_core::ImuData &data_minus, const ov_core::ImuData &data_plus,
                                     Eigen::Matrix<double, 27, 27> &F, Eigen::Matrix<double, 27, 15> &Fw) {

  // Time elapsed over interval
  double dt = data_plus.timestamp - data_minus.timestamp;

  // Corrected imu measurements
  Eigen::Matrix<double, 3, 1> w_hat1 = data_minus.wm - state->_imu->bias_g();
  Eigen::Matrix<double, 3, 1> a_hat1 = data_minus.am - state->_imu->bias_a();
  Eigen::Matrix<double, 3, 1> w_hat2 = data_plus.wm - state->_imu->bias_g();
  Eigen::Matrix<double, 3, 1> a_hat2 = data_plus.am - state->_imu->bias_a();
  Eigen::Vector3d w_hat;
  Eigen::Vector3d a_hat;

  // If we are averaging the IMU, then do so
  if (state->_options.imu_avg) {
    w_hat = .5 * (w_hat1 + w_hat2);
    a_hat = .5 * (a_hat1 + a_hat2);
  } else {
    w_hat = w_hat2;
    a_hat = a_hat2;
  }

  // Compute the new state mean value
  Eigen::Vector4d new_q;
  Eigen::Vector3d new_p, new_v;
  Eigen::Vector3d new_offset_upperLeft, new_offset_bottomLeft, new_offset_bottomRight, new_offset_upperRight;

  // calculated valuables needed for propagation state->_imu->Rot()
  wc_vec = state->c_RotMtrx_i * w_hat;
  vc_vec = state->c_RotMtrx_i * (state->_imu->vel() + skew_x(w_hat) * state->i_tVec_i2c);
  muc_vec = state->c_RotMtrx_i * state->_imu->Rot().transpose() * _muw_vec; // normal vector of the ground expressed in camera frame
  dc = (state->_imu->Rot() * (state->_imu->pos() + state->i_tVec_i2c))(2, 0); // scalar

  pt_upperLeft   = state->cam_upperLeft_pt_xy1   + state->_offset_upperLeft->value();  
  pt_bottomLeft  = state->cam_bottomLeft_pt_xy1  + state->_offset_bottomLeft->value(); 
  pt_bottomRight = state->cam_bottomRight_pt_xy1 + state->_offset_bottomRight->value();
  pt_upperRight  = state->cam_upperRight_pt_xy1  + state->_offset_upperRight->value(); 

  predict_mean_discrete(state, dt, w_hat, a_hat, new_p, new_q, new_v, new_offset_upperLeft, new_offset_bottomLeft, new_offset_bottomRight, new_offset_upperRight);

  F.block(state->p_id, state->p_id, 3, 3) = _I33 - dt * skew_x(w_hat);
  F.block(state->p_id, state->v_id, 3, 3) = dt * _I33;
  F.block(state->p_id, state->bg_id, 3, 3) = - dt * skew_x(state->_imu->pos());

  F.block(state->q_id, state->q_id, 3, 3) = Ham_quat_2_Rot(rotVec_2_Ham_quat(w_hat * dt)).transpose();
  F.block(state->q_id, state->bg_id, 3, 3) = - dt * Jr_theta(w_hat * dt);
  
  F.block(state->v_id, state->q_id, 3, 3) = dt * skew_x(state->_imu->Rot().transpose() * _gravity);
  F.block(state->v_id, state->v_id, 3, 3) = _I33 - dt * skew_x(w_hat); 
  F.block(state->v_id, state->ba_id, 3, 3) = - dt * _I33;
  F.block(state->v_id, state->bg_id, 3, 3) = - dt * skew_x(state->_imu->vel());

  F.block(state->ba_id, state->ba_id, 3, 3) = _I33;
  F.block(state->bg_id, state->bg_id, 3, 3) = _I33;

  // 4pt offset related
  double scalar = _ezT * vc_vec;
  scalar = scalar / dc;

  // corner upperLeft
  Eigen::Matrix<double, 3, 3> J_df_pt_upperLeft = skew_x(wc_vec) + vc_vec * muc_vec.transpose() / dc 
                                                  - _ezT * skew_x(wc_vec) * pt_upperLeft * _I33 - pt_upperLeft * _ezT * skew_x(wc_vec) 
                                                  - scalar * (muc_vec.transpose() * pt_upperLeft * _I33 + pt_upperLeft * muc_vec.transpose()); //  
  Eigen::Matrix<double, 3, 3> common_upperLeft = _I33 - pt_upperLeft * _ezT;
  Eigen::Matrix<double, 3, 1> J_df_dc_upperLeft = 1.0 / dc / dc * muc_vec.transpose() * pt_upperLeft * (- common_upperLeft) * vc_vec;
  Eigen::Matrix<double, 3, 3> J_df_vc_upperLeft = 1.0 / dc * muc_vec.transpose() * pt_upperLeft * common_upperLeft;
  Eigen::Matrix<double, 3, 3> J_df_muc_upperLeft = 1.0 / dc * common_upperLeft * vc_vec * pt_upperLeft.transpose();
  Eigen::Matrix<double, 3, 3> J_df_wc_upperLeft = - common_upperLeft * skew_x(pt_upperLeft);

  // corner bottomLeft
  Eigen::Matrix<double, 3, 3> J_df_pt_bottomLeft = skew_x(wc_vec) + vc_vec * muc_vec.transpose() / dc 
                                                   - _ezT * skew_x(wc_vec) * pt_bottomLeft * _I33 - pt_bottomLeft * _ezT * skew_x(wc_vec) 
                                                   - scalar * (muc_vec.transpose() * pt_bottomLeft * _I33 + pt_bottomLeft * muc_vec.transpose());
  Eigen::Matrix<double, 3, 3> common_bottomLeft = _I33 - pt_bottomLeft * _ezT;
  Eigen::Matrix<double, 3, 1> J_df_dc_bottomLeft = 1.0 / dc / dc * muc_vec.transpose() * pt_bottomLeft * (- common_bottomLeft) * vc_vec;
  Eigen::Matrix<double, 3, 3> J_df_vc_bottomLeft = 1.0 / dc * muc_vec.transpose() * pt_bottomLeft * common_bottomLeft;
  Eigen::Matrix<double, 3, 3> J_df_muc_bottomLeft = 1.0 / dc * common_bottomLeft * vc_vec * pt_bottomLeft.transpose();
  Eigen::Matrix<double, 3, 3> J_df_wc_bottomLeft = - common_bottomLeft * skew_x(pt_bottomLeft);

  // corner bottomRight
  Eigen::Matrix<double, 3, 3> J_df_pt_bottomRight = skew_x(wc_vec) + vc_vec * muc_vec.transpose() / dc 
                                                    - _ezT * skew_x(wc_vec) * pt_bottomRight * _I33 - pt_bottomRight * _ezT * skew_x(wc_vec) 
                                                    - scalar * (muc_vec.transpose() * pt_bottomRight * _I33 + pt_bottomRight * muc_vec.transpose());
  Eigen::Matrix<double, 3, 3> common_bottomRight = _I33 - pt_bottomRight * _ezT;
  Eigen::Matrix<double, 3, 1> J_df_dc_bottomRight = 1.0 / dc / dc * muc_vec.transpose() * pt_bottomRight * (- common_bottomRight) * vc_vec;
  Eigen::Matrix<double, 3, 3> J_df_vc_bottomRight = 1.0 / dc * muc_vec.transpose() * pt_bottomRight * common_bottomRight;
  Eigen::Matrix<double, 3, 3> J_df_muc_bottomRight = 1.0 / dc * common_bottomRight * vc_vec * pt_bottomRight.transpose();
  Eigen::Matrix<double, 3, 3> J_df_wc_bottomRight = - common_bottomRight * skew_x(pt_bottomRight);

  // corner upperRight
  Eigen::Matrix<double, 3, 3>  J_df_pt_upperRight = skew_x(wc_vec) + vc_vec * muc_vec.transpose() / dc 
                                                    - _ezT * skew_x(wc_vec) * pt_upperRight * _I33 - pt_upperRight * _ezT * skew_x(wc_vec) 
                                                    - scalar * (muc_vec.transpose() * pt_upperRight * _I33 + pt_upperRight * muc_vec.transpose());
  Eigen::Matrix<double, 3, 3> common_upperRight = _I33 - pt_upperRight * _ezT;
  Eigen::Matrix<double, 3, 1> J_df_dc_upperRight = 1.0 / dc / dc * muc_vec.transpose() * pt_upperRight * (- common_upperRight) * vc_vec;
  Eigen::Matrix<double, 3, 3> J_df_vc_upperRight = 1.0 / dc * muc_vec.transpose() * pt_upperRight * common_upperRight;
  Eigen::Matrix<double, 3, 3> J_df_muc_upperRight = 1.0 / dc * common_upperRight * vc_vec * pt_upperRight.transpose();
  Eigen::Matrix<double, 3, 3> J_df_wc_upperRight = - common_upperRight * skew_x(pt_upperRight);

  // same for 4 corners
  Eigen::Matrix<double, 3, 3> J_f_df = - dt * _I33;
  Eigen::Matrix<double, 1, 3> J_dc_p = _ezT * state->_imu->Rot();
  Eigen::Matrix<double, 1, 3> J_dc_q = _ezT * (- state->_imu->Rot() * skew_x(state->_imu->pos() + state->i_tVec_i2c));
  Eigen::Matrix<double, 3, 3> J_muc_q = state->c_RotMtrx_i * skew_x(state->_imu->Rot().transpose() * _muw_vec);

  // corner upperLeft
  F.block(state->pt_ul_id, state->p_id, 3, 3)     = J_f_df * J_df_dc_upperLeft * J_dc_p;
  F.block(state->pt_ul_id, state->q_id, 3, 3)     = J_f_df * (J_df_dc_upperLeft * J_dc_q + J_df_muc_upperLeft * J_muc_q);
  F.block(state->pt_ul_id, state->v_id, 3, 3)     = J_f_df * J_df_vc_upperLeft * _J_vc_v;
  F.block(state->pt_ul_id, state->bg_id, 3, 3)    = J_f_df * (J_df_vc_upperLeft * _J_vc_bw + J_df_wc_upperLeft * _J_wc_bw);
  F.block(state->pt_ul_id, state->pt_ul_id, 3, 3) = _I33 + J_f_df * J_df_pt_upperLeft;
  // corner bottomLeft
  F.block(state->pt_bl_id, state->p_id, 3, 3)     = J_f_df * J_df_dc_bottomLeft * J_dc_p;
  F.block(state->pt_bl_id, state->q_id, 3, 3)     = J_f_df * (J_df_dc_bottomLeft * J_dc_q + J_df_muc_bottomLeft * J_muc_q);
  F.block(state->pt_bl_id, state->v_id, 3, 3)     = J_f_df * J_df_vc_bottomLeft * _J_vc_v;
  F.block(state->pt_bl_id, state->bg_id, 3, 3)    = J_f_df * (J_df_vc_bottomLeft * _J_vc_bw + J_df_wc_bottomLeft * _J_wc_bw);
  F.block(state->pt_bl_id, state->pt_bl_id, 3, 3) = _I33 + J_f_df * J_df_pt_bottomLeft;
  // corner bottomRight
  F.block(state->pt_br_id, state->p_id, 3, 3)     = J_f_df * J_df_dc_bottomRight * J_dc_p;
  F.block(state->pt_br_id, state->q_id, 3, 3)     = J_f_df * (J_df_dc_bottomRight * J_dc_q + J_df_muc_bottomRight * J_muc_q);
  F.block(state->pt_br_id, state->v_id, 3, 3)     = J_f_df * J_df_vc_bottomRight * _J_vc_v;
  F.block(state->pt_br_id, state->bg_id, 3, 3)    = J_f_df * (J_df_vc_bottomRight * _J_vc_bw + J_df_wc_bottomRight * _J_wc_bw);
  F.block(state->pt_br_id, state->pt_br_id, 3, 3) = _I33 + J_f_df * J_df_pt_bottomRight;
  // corner upperRight
  F.block(state->pt_ur_id, state->p_id, 3, 3)     = J_f_df * J_df_dc_upperRight * J_dc_p;
  F.block(state->pt_ur_id, state->q_id, 3, 3)     = J_f_df * (J_df_dc_upperRight * J_dc_q + J_df_muc_upperRight * J_muc_q);
  F.block(state->pt_ur_id, state->v_id, 3, 3)     = J_f_df * J_df_vc_upperRight * _J_vc_v;
  F.block(state->pt_ur_id, state->bg_id, 3, 3)    = J_f_df * (J_df_vc_upperRight * _J_vc_bw + J_df_wc_upperRight * _J_wc_bw);
  F.block(state->pt_ur_id, state->pt_ur_id, 3, 3) = _I33 + J_f_df * J_df_pt_upperRight;

  Fw.block(state->p_id, 0, 3, 3) = - F.block(state->p_id, state->bg_id, 3, 3);
  Fw.block(state->p_id, 12, 3, 3) = F.block(state->p_id, state->v_id, 3, 3);
  Fw.block(state->q_id, 0, 3, 3) = - F.block(state->q_id, state->bg_id, 3, 3);
  Fw.block(state->v_id, 0, 3, 3) = - F.block(state->v_id, state->bg_id, 3, 3);
  Fw.block(state->v_id, 3, 3, 3) = Fw.block(state->p_id, 12, 3, 3);
  Fw.block(state->ba_id, 6, 3, 3) = Fw.block(state->p_id, 12, 3, 3);
  Fw.block(state->bg_id, 9, 3, 3) = Fw.block(state->p_id, 12, 3, 3);

  Fw.block(state->pt_ul_id, 0, 3, 3) = - F.block(state->pt_ul_id, state->bg_id, 3, 3);
  Fw.block(state->pt_bl_id, 0, 3, 3) = - F.block(state->pt_bl_id, state->bg_id, 3, 3); 
  Fw.block(state->pt_br_id, 0, 3, 3) = - F.block(state->pt_br_id, state->bg_id, 3, 3);
  Fw.block(state->pt_ur_id, 0, 3, 3) = - F.block(state->pt_ur_id, state->bg_id, 3, 3); 

  // Now replace imu estimate with propagated values
  Eigen::Matrix<double, 16, 1> imu_x = state->_imu->value();
  imu_x.block(0, 0, 3, 1) = new_p;
  imu_x.block(3, 0, 4, 1) = new_q;
  imu_x.block(7, 0, 3, 1) = new_v;
  state->_imu->set_value(imu_x);
  // state->_imu->set_fej(imu_x);
  state->_offset_upperLeft  ->set_value(new_offset_upperLeft);
  state->_offset_bottomLeft ->set_value(new_offset_bottomLeft);
  state->_offset_bottomRight->set_value(new_offset_bottomRight);
  state->_offset_upperRight ->set_value(new_offset_upperRight);

}

// CUAHN uses this one! 
void Propagator::predict_mean_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                       Eigen::Vector3d &new_p, Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, 
                                       Eigen::Vector3d &new_offset_upperLeft, Eigen::Vector3d &new_offset_bottomLeft, 
                                       Eigen::Vector3d &new_offset_bottomRight, Eigen::Vector3d &new_offset_upperRight) {

  // Pre-compute things
  new_q = quatnorm(Ham_quat_update(w_hat*dt) * state->_imu->quat());

  // Velocity: just the acceleration in the local frame, minus global gravity
  new_v = state->_imu->vel() + dt * (- skew_x(w_hat) * state->_imu->vel() + a_hat + state->_imu->Rot().transpose() * _gravity); // _gravity=[0;0;-9.81]

  // Position: just velocity times dt, with the acceleration integrated twice
  new_p = state->_imu->pos() + dt * (- skew_x(w_hat) * state->_imu->pos() + state->_imu->vel());

  // update 4pt offset prediction
  Eigen::Matrix<double, 3, 3> H = skew_x(wc_vec) + vc_vec * muc_vec.transpose() / dc;

  new_offset_upperLeft   = state->_offset_upperLeft->value()   + dt * (- (_I33 -   pt_upperLeft * _ezT) * H * pt_upperLeft);
  new_offset_bottomLeft  = state->_offset_bottomLeft->value()  + dt * (- (_I33 -  pt_bottomLeft * _ezT) * H * pt_bottomLeft); 
  new_offset_bottomRight = state->_offset_bottomRight->value() + dt * (- (_I33 - pt_bottomRight * _ezT) * H * pt_bottomRight);
  new_offset_upperRight  = state->_offset_upperRight->value()  + dt * (- (_I33 -  pt_upperRight * _ezT) * H * pt_upperRight);

}