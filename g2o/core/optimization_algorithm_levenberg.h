#pragma once

#include "optimization_algorithm.h"

namespace g2o {
    class OptimizationAlgorithmLevenberg : public OptimizationAlgorithm<> {
    public:
        using OptimizationAlgorithm::OptimizationAlgorithm;
    };
}
