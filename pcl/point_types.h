#pragma once

// PCL Point Types Stub
#include "point_cloud.h"

namespace pcl {

// Additional point types that might be used
struct PointNormal {
    float x, y, z;
    float normal_x, normal_y, normal_z;
    float curvature;
    PointNormal() : x(0), y(0), z(0), normal_x(0), normal_y(0), normal_z(0), curvature(0) {}
};

struct PointXYZI {
    float x, y, z;
    float intensity;
    PointXYZI() : x(0), y(0), z(0), intensity(0) {}
};

// Additional convenience typedefs
typedef PointCloud<PointNormal> PointCloudNormal;
typedef PointCloud<PointXYZI> PointCloudXYZI;

} // namespace pcl
