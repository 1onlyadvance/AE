#pragma once

#include "../point_cloud.h"

namespace pcl
{
  template <typename PointInT, typename PointOutT>
  class Feature
  {
    public:
      using PointCloudIn = PointCloud<PointInT>;
      using PointCloudInPtr = typename PointCloudIn::Ptr;

      Feature () = default;
      virtual ~Feature () = default;

      virtual void
      setInputCloud (const PointCloudInPtr &cloud)
      {
        input_ = cloud;
      }

    protected:
      PointCloudInPtr input_;
  };
}
