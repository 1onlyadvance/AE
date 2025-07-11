#pragma once

#include "../../core/sparse_optimizer.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include "../../core/base_vertex.h"
#include "se3quat.h"

namespace g2o {

class VertexSE3 : public BaseVertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSE3() = default;
    virtual ~VertexSE3() = default;
    
    void setToOriginImpl() override {}
    void oplusImpl(const double* update) override {}
    void setId(int) {}
    void setFixed(bool) {}
    
    void setEstimate(const Eigen::Isometry3d& pose) { pose_ = pose; }
    const Eigen::Isometry3d& estimate() const { return pose_; }
    
private:
    Eigen::Isometry3d pose_ = Eigen::Isometry3d::Identity();
};

} // namespace g2o
