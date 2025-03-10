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

#ifndef OV_TYPE_TYPE_HAMQUAT_H
#define OV_TYPE_TYPE_HAMQUAT_H

#include "Type.h"
#include "utils/quat_ops.h"

namespace ov_type {

/**
 * @brief Derived Type class that implements Hamilton quaternion
 *
 */
class HamQuat : public Type {

public:
  HamQuat() : Type(3) { // [qw; qx; qy; qz]
    Eigen::Vector4d q0 = Eigen::Vector4d::Zero();
    q0(0) = 1.0;
    set_value_internal(q0);
    // set_fej_internal(q0);
  }

  ~HamQuat() {}

  /**
   * @brief same as the quaternion.update of the Python implementation 
   *
   * @param rot_vec Axis-angle representation of the perturbing quaternion
   */
  void update(const Eigen::VectorXd &rot_vec) override {

    assert(rot_vec.rows() == _size);

    // Update estimate and recompute R
    set_value(ov_core::quatnorm(ov_core::Ham_quat_update(rot_vec) * _value));
  }

  /**
   * @brief Sets the value of the estimate and recomputes the internal rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

  /**
   * @brief Sets the fej value and recomputes the fej rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_fej(const Eigen::MatrixXd &new_value) override { set_fej_internal(new_value); }

  std::shared_ptr<Type> clone() override {
    auto Clone = std::shared_ptr<HamQuat>(new HamQuat());
    Clone->set_value(value());
    Clone->set_fej(fej());
    return Clone;
  }

  /// Rotation access
  Eigen::Matrix<double, 3, 3> Rot() const { return _R; }

  /// FEJ Rotation access
  Eigen::Matrix<double, 3, 3> Rot_fej() const { return _Rfej; }

protected:
  // Stores the rotation
  Eigen::Matrix<double, 3, 3> _R;

  // Stores the first-estimate rotation
  Eigen::Matrix<double, 3, 3> _Rfej;

  /**
   * @brief Sets the value of the estimate and recomputes the internal rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_value_internal(const Eigen::MatrixXd &new_value) {

    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _value = new_value;

    // compute associated rotation
    _R = ov_core::Ham_quat_2_Rot(new_value);
  }

  /**
   * @brief Sets the fej value and recomputes the fej rotation matrix
   * @param new_value New value for the quaternion estimate
   */
  void set_fej_internal(const Eigen::MatrixXd &new_value) {

    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _fej = new_value;

    // compute associated rotation
    _Rfej = ov_core::Ham_quat_2_Rot(new_value);
  }
};

} // namespace ov_type

#endif // OV_TYPE_TYPE_HAMQUAT_H
