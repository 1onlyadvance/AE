/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Helper file for cuBLAS integration
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace ares::countermeasures {

// Error checking macro for cuBLAS
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + std::to_string(static_cast<int>(status))); \
    } \
} while(0)

// Helper to convert cuBLAS error to CUDA error
inline cudaError_t cublasToCudaError(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return cudaSuccess;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return cudaErrorMemoryAllocation;
        case CUBLAS_STATUS_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return cudaErrorInvalidDeviceFunction;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return cudaErrorLaunchFailure;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return cudaErrorUnknown;
        default:
            return cudaErrorUnknown;
    }
}

} // namespace ares::countermeasures
