#pragma once

#include "../core/sparse_optimizer.h"
#include "block_solver_traits.h"

namespace g2o {

class BlockSolver {
public:
    BlockSolver() = default;
    virtual ~BlockSolver() = default;
};

} // namespace g2o
