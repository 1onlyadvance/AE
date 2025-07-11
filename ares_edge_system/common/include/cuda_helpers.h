#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <iostream>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t cudaStatus = call; \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(cudaStatus) << std::endl; \
        } \
    } while(0)

// Macro for checking kernel launch errors
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(cudaStatus) << std::endl; \
        } \
        cudaStatus = cudaDeviceSynchronize(); \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA synchronize error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(cudaStatus) << std::endl; \
        } \
    } while(0)

// Helper function to get device properties
inline void printDeviceProperties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
}

// Helper function to calculate grid and block dimensions
inline dim3 calculateGridDim(int totalThreads, int blockSize = 256) {
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    return dim3(gridSize);
}

#endif // CUDA_HELPERS_H
