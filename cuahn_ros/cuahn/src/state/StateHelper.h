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

#ifndef CUAHN_STATE_HELPER_H
#define CUAHN_STATE_HELPER_H

#include "State.h"
#include "types/Landmark.h"
#include "utils/colors.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;

namespace cuahn {

/**
 * @brief Helper which manipulates the State and its covariance.
 *
 * In general, this class has all the core logic for an Extended Kalman Filter (EKF)-based system.
 * This has all functions that change the covariance along with addition and removing elements from the state.
 * All functions here are static, and thus are self-contained so that in the future multiple states could be tracked and updated.
 * We recommend you look directly at the code for this class for clarity on what exactly we are doing in each and the matching documentation
 * pages.
 */
class StateHelper {

public:

  // CUAHN propagate the Cov matrix
  static void propagate_Cov(std::shared_ptr<State> state, const Eigen::MatrixXd &F, const Eigen::MatrixXd &Fw, const Eigen::MatrixXd &Q);

  static void initialize_Cov(std::shared_ptr<State> state, const Eigen::Vector4d &q_I0toW);

private:
  /**
   * All function in this class should be static.
   * Thus an instance of this class cannot be created.
   */
  StateHelper() {}
};

} // namespace cuahn

#endif // CUAHN_STATE_HELPER_H
