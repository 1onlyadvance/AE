#pragma once

namespace g2o {
class BlockSolverTraitsBase {};
template<int, int> class BlockSolverTraits : public BlockSolverTraitsBase {
public:
    using PoseMatrixType = float;
};
}
