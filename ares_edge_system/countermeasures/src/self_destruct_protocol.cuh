/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file defines the SelfDestructProtocol header for the ARES Edge System
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "destruct_mode.h"

namespace ares::countermeasures {

// Forward declaration for the CUDA kernel
__global__ void initialize_chaos_random_states(curandState* states, uint32_t num_states, uint64_t seed);

// Self-destruct configuration
constexpr uint32_t AUTH_KEY_SIZE = 256;  // bits
constexpr uint32_t VERIFICATION_STAGES = 5;
constexpr uint32_t FAILSAFE_CODES = 3;

// Authentication state
struct AuthenticationState {
    uint8_t auth_keys[VERIFICATION_STAGES][AUTH_KEY_SIZE / 8];
    uint8_t current_stage;
    bool stages_verified[VERIFICATION_STAGES];
    uint64_t auth_timestamp_ns;
    uint32_t failed_attempts;
    bool locked_out;
};

// Countdown state
struct CountdownState {
    float remaining_ms;
    float initial_ms;
    uint32_t abort_codes_entered;
    uint8_t abort_code_hash[32];  // SHA-256
    bool countdown_active;
    bool abort_window_open;
    uint64_t start_timestamp_ns;
};

class SelfDestructProtocol {
private:
    // Device memory
    thrust::device_vector<uint8_t> d_secure_memory;
    thrust::device_vector<uint32_t> d_memory_regions;
    thrust::device_vector<AuthenticationState> d_auth_state;
    thrust::device_vector<CountdownState> d_countdown_state;
    thrust::device_vector<float> d_em_waveform;
    thrust::device_vector<curandState_t> d_rand_states;
    
    // CUDA resources
    cudaStream_t destruct_stream;
    
    // Control state
    std::atomic<DestructMode> destruct_mode{DestructMode::DATA_WIPE};
    std::atomic<bool> armed{false};
    std::mutex control_mutex;
    
public:
    SelfDestructProtocol();
    ~SelfDestructProtocol();

    void set_destruct_mode(DestructMode mode);
    DestructMode get_destruct_mode() const;
    bool is_armed() const;
    void arm_system();
    void disarm_system();
    void execute_destruction();
    void emergency_maximum_chaos();
};

} // namespace ares::countermeasures
