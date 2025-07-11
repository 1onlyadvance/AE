#pragma once

// PCL (Point Cloud Library) Stub
// This is a minimal stub for the Point Cloud Library

#include <vector>
#include <memory>

namespace pcl {

struct PointXYZ {
    float x, y, z;
    PointXYZ() : x(0), y(0), z(0) {}
    PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct PointXYZRGB {
    float x, y, z;
    uint32_t rgb;
    PointXYZRGB() : x(0), y(0), z(0), rgb(0) {}
};

template<typename PointT>
class PointCloud {
public:
    typedef std::shared_ptr<PointCloud<PointT>> Ptr;
    typedef std::shared_ptr<const PointCloud<PointT>> ConstPtr;
    
    std::vector<PointT> points;
    uint32_t width = 0;
    uint32_t height = 1;
    bool is_dense = true;
    
    PointCloud() = default;
    
    size_t size() const { return points.size(); }
    void resize(size_t size) { points.resize(size); }
    void clear() { points.clear(); }
    
    PointT& operator[](size_t idx) { return points[idx]; }
    const PointT& operator[](size_t idx) const { return points[idx]; }
    
    typename std::vector<PointT>::iterator begin() { return points.begin(); }
    typename std::vector<PointT>::iterator end() { return points.end(); }
    typename std::vector<PointT>::const_iterator begin() const { return points.begin(); }
    typename std::vector<PointT>::const_iterator end() const { return points.end(); }
    
    PointCloud& operator+=(const PointCloud&) { return *this; }
};

// Convenience typedefs
typedef PointCloud<PointXYZ> PointCloudXYZ;
typedef PointCloud<PointXYZRGB> PointCloudXYZRGB;

} // namespace pcl
