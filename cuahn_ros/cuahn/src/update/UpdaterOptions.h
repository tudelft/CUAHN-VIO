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

#ifndef CUAHN_UPDATER_OPTIONS_H
#define CUAHN_UPDATER_OPTIONS_H

namespace cuahn {

/**
 * @brief Struct which stores general updater options
 */
struct UpdaterOptions {

  double K_net_Cov = 10.0;

  /// Nice print function of what parameters we have loaded
  void print() {
  }
};

} // namespace cuahn

#endif // CUAHN_UPDATER_OPTIONS_H