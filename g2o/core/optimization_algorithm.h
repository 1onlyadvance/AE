#pragma once

#include "optimizable_graph.h"

namespace g2o {
    template <typename... Args>
    class OptimizationAlgorithm {
    public:
        explicit OptimizationAlgorithm(std::unique_ptr<Solver> solver) {}
    };
}
