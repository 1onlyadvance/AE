#pragma once

#include "../point_cloud.h"

namespace pcl {
namespace search {

template<typename PointT>
class Search {
public:
    typedef std::shared_ptr<Search<PointT>> Ptr;
    virtual void setInputCloud(const typename PointCloud<PointT>::ConstPtr& cloud) = 0;
};

template<typename PointT>
class KdTree : public Search<PointT> {
public:
    typedef std::shared_ptr<KdTree<PointT>> Ptr;
    
    KdTree() = default;
    
    void setInputCloud(const typename PointCloud<PointT>::ConstPtr& cloud) override {
        input_cloud_ = cloud;
    }

private:
    typename PointCloud<PointT>::ConstPtr input_cloud_;
};

} // namespace search
} // namespace pcl
