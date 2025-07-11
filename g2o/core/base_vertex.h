#pragma once

#include "../core/optimizable_graph.h"

namespace g2o {
    class BaseVertex : public OptimizableGraph::Vertex {
    public:
        virtual void setToOriginImpl() = 0;
        virtual void oplusImpl(const double* update) = 0;
    };
}
