#pragma once

#include "feature.h"

namespace pcl
{
  template <typename PointInT, typename PointNT, typename PointOutT>
  class FPFHEstimation : public Feature<PointInT, PointOutT>
  {
    public:
      FPFHEstimation () = default;
      ~FPFHEstimation () = default;

      void compute (PointCloud<PointOutT> &output)
      {
        // Stub implementation
        output.points.resize (this->input_->size ());
        for (size_t i = 0; i < output.points.size (); ++i)
        {
          // Fill with some dummy data
          for (size_t j = 0; j < sizeof(output.points[i].histogram) / sizeof(float); ++j)
          {
            output.points[i].histogram[j] = static_cast<float> (j);
          }
        }
      }
  };
}
