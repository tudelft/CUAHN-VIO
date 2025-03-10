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
 * You should have received a copy of the GNU GeneralEckenhoff Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CUAHN_STATE_OPTIONS_H
#define CUAHN_STATE_OPTIONS_H

#include "types/LandmarkRepresentation.h"
#include <climits>

using namespace ov_type;

namespace cuahn {

/**
 * @brief Struct which stores all our filter options
 */
struct StateOptions {

  /// Bool to determine whether or not use imu message averaging
  bool imu_avg = true;

  /// Number of distinct cameras that we will observe features in
  int num_cameras = 1;

  /// Nice print function of what parameters we have loaded
  void print() {
    printf("\t- use_imuavg: %d\n", imu_avg);
    printf("\t- max_cameras: %d\n", num_cameras);
  }
};

} // namespace cuahn

#endif // CUAHN_STATE_OPTIONS_H