/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file defines the ChaosInductionEngine header for the ARES Edge System
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <atomic>

namespace ares::countermeasures {

// Chaos modes for the system
enum class ChaosMode : uint8_t {
    DISABLED = 0,
    SIGNATURE_SWAPPING = 1,
    COORDINATED_MISDIRECTION = 2,
    AGGRESSIVE_DECEPTION = 3,
    MAXIMUM_ENTROPY = 4
};

// Forward declaration for the CUDA kernel
__global__ void initialize_chaos_random_states(curandState* states, uint32_t num_states, uint64_t seed);

class ChaosInductionEngine {
private:
    thrust::device_vector<curandState> d_random_states;
    cudaStream_t chaos_stream;
    std::atomic<ChaosMode> chaos_mode{ChaosMode::DISABLED};
    std::atomic<float> chaos_intensity{0.0f};
    bool initialized{false};

public:
    ChaosInductionEngine();
    ~ChaosInductionEngine();

    void initialize(uint32_t num_states = 1024, uint64_t seed = 0);
    void set_chaos_mode(ChaosMode mode);
    void set_chaos_intensity(float intensity);
    void emergency_maximum_chaos();
    ChaosMode get_chaos_mode() const;
    float get_chaos_intensity() const;
};

} // namespace ares::countermeasures
