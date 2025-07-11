#pragma once

#include "../core/optimizable_graph.h"

namespace g2o {
    class BaseEdge : public OptimizableGraph::Edge {
    public:
        virtual void computeError() = 0;
        virtual void linearizeOplus() = 0;
    };
}
