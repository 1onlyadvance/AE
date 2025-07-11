#pragma once

#include <Eigen/Core>

namespace g2o {
class SE3Quat {
public:
    SE3Quat() = default;
    SE3Quat(const Eigen::Matrix3d&, const Eigen::Vector3d&) {}
    Eigen::Matrix3d rotation() const { return Eigen::Matrix3d::Identity(); }
    Eigen::Vector3d translation() const { return Eigen::Vector3d::Zero(); }
    // Conversion operator stub
    operator Eigen::Isometry3d() const { return Eigen::Isometry3d::Identity(); }
};
}
