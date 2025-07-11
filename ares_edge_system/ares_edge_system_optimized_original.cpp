#include <cstddef> // for size_t

// CUDA types for non-CUDA build
typedef enum cudaError {
    cudaSuccess = 0
} cudaError_t;

// Add to cew/include/cew_adaptive_jamming.h in the AdaptiveJammingModule class
cudaError_t processSpectrumWaterfall(float* buffer, size_t size) {
    // Stub implementation
    return cudaSuccess;
}
