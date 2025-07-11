#pragma once

#include "vertex_se3.h"
#include "../../core/sparse_optimizer.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include "../../core/base_edge.h"
#include "se3quat.h"

namespace g2o {

class EdgeSE3 : public BaseEdge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3() = default;
    virtual ~EdgeSE3() = default;
    
    void computeError() override {}
    void linearizeOplus() override {}
    void setId(int) {}
    
    void setMeasurement(const Eigen::Isometry3d& measurement) { measurement_ = measurement; }
    const Eigen::Isometry3d& measurement() const { return measurement_; }
    
    void setInformation(const Eigen::Matrix<double, 6, 6>& information) { information_ = information; }
    const Eigen::Matrix<double, 6, 6>& information() const { return information_; }
    
    template<typename T> void setVertex(int, T*) {}
    
private:
    Eigen::Isometry3d measurement_ = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> information_ = Eigen::Matrix<double, 6, 6>::Identity();
};

} // namespace g2o
