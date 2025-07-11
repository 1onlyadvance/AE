#pragma once

#include "../point_cloud.h"

namespace pcl {

struct PointXYZRGB {};

template<typename PointSource, typename PointTarget>
class Registration {
public:
    Registration() {}
    void setInputSource(void*) {}
    void setInputTarget(void*) {}
};

template<typename PointSource, typename PointTarget>
class IterativeClosestPoint : public Registration<PointSource, PointTarget> {
public:
    IterativeClosestPoint() : max_iterations_(10), transformation_epsilon_(1e-8), max_correspondence_distance_(0.1) {}
    
    void setInputSource(const typename PointCloud<PointSource>::Ptr& cloud) {
        source_cloud_ = cloud;
    }
    
    void setInputTarget(const typename PointCloud<PointTarget>::Ptr& cloud) {
        target_cloud_ = cloud;
    }
    
    void setMaxCorrespondenceDistance(double dist) {
        max_correspondence_distance_ = dist;
    }
    
    void setMaximumIterations(int iter) {
        max_iterations_ = iter;
    }
    
    void setTransformationEpsilon(double epsilon) {
        transformation_epsilon_ = epsilon;
    }
    
    void align(PointCloud<PointSource>& output) {
        // Stub: Just copy source to output
        if (source_cloud_) {
            output = *source_cloud_;
        }
    }
    
    bool hasConverged() const { return true; }
    
    Eigen::Matrix4f getFinalTransformation() const {
        return Eigen::Matrix4f::Identity();
    }

private:
    typename PointCloud<PointSource>::Ptr source_cloud_;
    typename PointCloud<PointTarget>::Ptr target_cloud_;
    int max_iterations_;
    double transformation_epsilon_;
    double max_correspondence_distance_;
};

} // namespace pcl
