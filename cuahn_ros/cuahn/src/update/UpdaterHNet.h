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

#ifndef CUAHN_UPDATER_HNET_H
#define CUAHN_UPDATER_HNET_H

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/quat_ops.h"
#include <Eigen/Eigen>

#include "UpdaterOptions.h"


namespace cuahn {

/**
 * @brief Will compute the system for our sparse features and update the filter.
 *
 * This class is responsible for computing the entire linear system for all features that are going to be used in an update.
 * This follows the original MSCKF, where we first triangulate features, we then nullspace project the feature Jacobian.
 * After this we compress all the measurements to have an efficient update and update the state.
 */
class UpdaterHNet {

public:
  /**
   * @brief Default constructor for our MSCKF updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options Updater options (include measurement noise value)
   * @param feat_init_options Feature initializer options
   */
  UpdaterHNet(UpdaterOptions &options) : _options(options) {

    H = Eigen::Matrix<double, 8, 27>::Zero();
    H.block(0, 15, 2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
    H.block(2, 18, 2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
    H.block(4, 21, 2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
    H.block(6, 24, 2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
    
    Hn = Eigen::Matrix<double, 8, 8>::Identity();

  }

  /**
   * @brief Given network predictions, this will use them to update the state.
   *
   * @param state State of the filter
   * @param 
   */
  void update(std::shared_ptr<State> state, const Eigen::Matrix<double, 8, 1> &net_4pt_offset_mean,
              const Eigen::Matrix<double, 8, 8> &net_4pt_offset_Cov, const Eigen::Matrix<double, 8, 1> propagated_4pt_offset, bool update_offset);

protected:
  /// Options used during update
  UpdaterOptions _options;

  Eigen::Matrix<double, 8, 27> H;
  Eigen::Matrix<double, 8, 8> Hn;

};

} // namespace cuahn

#endif // CUAHN_UPDATER_HNET_H
