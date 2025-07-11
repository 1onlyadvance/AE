#!/bin/bash

# ARES Edge System Build Script
# Production-grade build with CUDA support

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}ARES Edge System Build Script${NC}"
echo -e "${BLUE}==================================${NC}"

# Function to check CUDA installation
check_cuda() {
    echo -e "${BLUE}Checking CUDA installation...${NC}"
    
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}ERROR: CUDA compiler (nvcc) not found!${NC}"
        echo "Please install CUDA Toolkit or add it to PATH"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}Found CUDA version: $CUDA_VERSION${NC}"
    
    # Set CUDA paths if not already set
    if [ -z "$CUDA_HOME" ]; then
        CUDA_HOME=/usr/local/cuda
        export CUDA_HOME
        echo -e "${BLUE}Set CUDA_HOME=$CUDA_HOME${NC}"
    fi
    
    # Add CUDA to library path
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
}

# Function to detect GPU architecture
detect_gpu_arch() {
    echo -e "${BLUE}Detecting GPU architecture...${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        echo -e "${GREEN}Detected GPU: $GPU_NAME${NC}"
        
        # Map GPU to compute capability
        case "$GPU_NAME" in
            *"V100"*) ARCH="70" ;;
            *"T4"*) ARCH="75" ;;
            *"A100"*) ARCH="80" ;;
            *"A10"*|*"A40"*) ARCH="86" ;;
            *"A30"*) ARCH="80" ;;
            *"RTX 3090"*|*"RTX 3080"*) ARCH="86" ;;
            *"RTX 4090"*|*"RTX 4080"*) ARCH="89" ;;
            *"H100"*) ARCH="90" ;;
            *) ARCH="70" ;; # Default to compute 7.0
        esac
        
        echo -e "${GREEN}Using compute capability: $ARCH${NC}"
    else
        echo -e "${RED}WARNING: nvidia-smi not found. Using default architecture.${NC}"
        ARCH="70"
    fi
}

# Clean build directory
clean_build() {
    echo -e "${BLUE}Cleaning build directory...${NC}"
    rm -rf build
    mkdir -p build
}

# Configure with CMake
configure_project() {
    echo -e "${BLUE}Configuring project with CMake...${NC}"
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}CMake configuration failed!${NC}"
        exit 1
    fi
}

# Build project
build_project() {
    echo -e "${BLUE}Building project...${NC}"
    cd build
    
    # Determine number of cores
    if [ -f /proc/cpuinfo ]; then
        CORES=$(grep -c ^processor /proc/cpuinfo)
    else
        CORES=4
    fi
    
    echo -e "${BLUE}Building with $CORES cores...${NC}"
    make -j$CORES
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
}

# Main execution
main() {
    # Parse arguments
    CLEAN=false
    ARCH=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN=true
                shift
                ;;
            --arch)
                ARCH="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--clean] [--arch <compute_capability>]"
                exit 1
                ;;
        esac
    done
    
    # Check CUDA
    check_cuda
    
    # Detect GPU if arch not specified
    if [ -z "$ARCH" ]; then
        detect_gpu_arch
    fi
    
    # Clean if requested
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    # Ensure build directory exists
    mkdir -p build
    
    # Configure and build
    configure_project
    build_project
    
    echo -e "${GREEN}==================================${NC}"
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo -e "${GREEN}Executable: build/ares_edge_system${NC}"
    echo -e "${GREEN}==================================${NC}"
}

# Run main function
main "$@"