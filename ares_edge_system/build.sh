#!/bin/bash
# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2024 DELFICTUS I/O LLC
# Patent Pending - Application #63/826,067

set -e

echo "=============================================="
echo "ARES Edge System - Build Script"
echo "DELFICTUS I/O LLC"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: CUDA not found. Please install CUDA 11.7+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CUDA found:${NC} $(nvcc --version | grep release)"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}WARNING: Docker not found. Using native build.${NC}"
    NATIVE_BUILD=1
else
    echo -e "${GREEN}✓ Docker found${NC}"
    NATIVE_BUILD=0
fi

# Build options
BUILD_TYPE=${1:-Release}
BUILD_DIR=${2:-build}

echo "Build Type: $BUILD_TYPE"
echo "Build Directory: $BUILD_DIR"

# Docker build (recommended)
if [ $NATIVE_BUILD -eq 0 ]; then
    echo -e "\n${GREEN}Building with Docker (Recommended)${NC}"
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -f docker/Dockerfile.cuda -t delfictus/ares-edge-system:latest .
    
    # Run build in container
    echo "Running build in container..."
    docker run --rm --gpus all \
        -v $(pwd):/workspace/ares_edge_system \
        delfictus/ares-edge-system:latest \
        bash -c "cd /workspace/ares_edge_system && \
                mkdir -p $BUILD_DIR && cd $BUILD_DIR && \
                cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE .. && \
                make -j\$(nproc)"
    
    echo -e "${GREEN}✓ Build complete!${NC}"
    echo "Binary location: $BUILD_DIR/ares_edge_system"
    
else
    # Native build
    echo -e "\n${YELLOW}Native Build${NC}"
    
    # Check dependencies
    echo "Checking dependencies..."
    
    MISSING_DEPS=()
    
    # Check for required libraries
    if ! pkg-config --exists libcrypto++; then
        MISSING_DEPS+=("libcrypto++-dev")
    fi
    
    if ! pkg-config --exists opencv4; then
        MISSING_DEPS+=("libopencv-dev")
    fi
    
    if ! pkg-config --exists eigen3; then
        MISSING_DEPS+=("libeigen3-dev")
    fi
    
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        echo -e "${RED}Missing dependencies:${NC}"
        printf '%s\n' "${MISSING_DEPS[@]}"
        echo -e "\nInstall with:"
        echo "sudo apt-get install ${MISSING_DEPS[*]}"
        exit 1
    fi
    
    # Create build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure
    echo "Configuring..."
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    
    # Build
    echo "Building..."
    make -j$(nproc)
    
    echo -e "${GREEN}✓ Build complete!${NC}"
    cd ..
fi

# Quick test
echo -e "\n${GREEN}Testing CUDA availability...${NC}"
cat > test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        return 0;
    }
    return 1;
}
EOF

nvcc -o test_cuda test_cuda.cu 2>/dev/null
if ./test_cuda; then
    echo -e "${GREEN}✓ CUDA test passed${NC}"
else
    echo -e "${RED}✗ CUDA test failed${NC}"
fi
rm -f test_cuda test_cuda.cu

echo -e "\n=============================================="
echo -e "${GREEN}Build Summary:${NC}"
echo "- Binary: $BUILD_DIR/ares_edge_system"
echo "- Config: $BUILD_TYPE"
echo "- Platform: $(uname -s) $(uname -m)"
echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"

echo -e "\n${GREEN}Next Steps:${NC}"
echo "1. Configure AI providers:"
echo "   export OPENAI_API_KEY='sk-...'"
echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "2. Run ARES:"
echo "   ./$BUILD_DIR/ares_edge_system --openai-key \$OPENAI_API_KEY"
echo ""
echo "3. For Unreal Engine integration:"
echo "   - Copy unreal/ARESEdgePlugin to your UE5 project's Plugins folder"
echo "   - Regenerate project files"
echo "   - Set ARESGameMode as default game mode"
echo "=============================================="