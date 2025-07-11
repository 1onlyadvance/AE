#pragma once

#include "../../core/sparse_optimizer.h"

namespace g2o {

class LinearSolverDense {
public:
    LinearSolverDense() = default;
    virtual ~LinearSolverDense() = default;
};

} // namespace g2o
