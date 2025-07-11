#pragma once

#include <memory>

namespace g2o {
    class HyperGraph {
    public:
        class Vertex {};
        class Edge {};
    };

    class OptimizableGraph : public HyperGraph {
    public:
        class Vertex : public HyperGraph::Vertex {
        public:
            void setId(int) {}
        };

        class Edge : public HyperGraph::Edge {
        public:
            void setVertex(size_t, Vertex*) {}
        };
    };

    class Solver {};
}
