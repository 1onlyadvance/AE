/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * Company: DELFICTUS I/O LLC
 * CAGE Code: 13H70
 * UEI: LXT3B9GMY4N8
 * Active DoD Contractor
 * 
 * Location: Los Angeles, California 90013 United States
 * 
 * This software contains trade secrets and proprietary information
 * of DELFICTUS I/O LLC. Unauthorized use, reproduction, or distribution
 * is strictly prohibited. This technology is subject to export controls
 * under ITAR and EAR regulations.
 * 
 * ARES Edge System™ - Autonomous Reconnaissance and Electronic Supremacy
 * 
 * WARNING: This system is designed for authorized U.S. Department of Defense
 * use only. Misuse may result in severe criminal and civil penalties.
 */

/**
 * @file cew_adaptive_jamming.cpp
 * @brief Implementation of Cognitive Electronic Warfare Adaptive Jamming Module
 */

#include "../include/cew_adaptive_jamming.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <chrono>
#include <cstring>

namespace ares::cew {

AdaptiveJammingModule::AdaptiveJammingModule() 
    : d_spectrum_buffer_(nullptr)
    , d_qtable_(nullptr)
    , d_waveform_bank_(nullptr)
    , compute_stream_(nullptr)
    , transfer_stream_(nullptr) {
    std::memset(&metrics_, 0, sizeof(metrics_));
}

AdaptiveJammingModule::~AdaptiveJammingModule() {
    if (d_spectrum_buffer_) cudaFree(d_spectrum_buffer_);
    if (d_qtable_) cudaFree(d_qtable_);
    if (d_waveform_bank_) cudaFree(d_waveform_bank_);
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_) cudaEventDestroy(stop_event_);
}

cudaError_t AdaptiveJammingModule::initialize(int device_id) {
    cudaError_t err;
    
    // Set device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return err;
    
    // Enable fastest GPU clocks
    err = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    if (err != cudaSuccess) return err;
    
    // Create high-priority streams
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    err = cudaStreamCreateWithPriority(&compute_stream_, cudaStreamNonBlocking, priority_high);
    if (err != cudaSuccess) return err;
    
    err = cudaStreamCreateWithPriority(&transfer_stream_, cudaStreamNonBlocking, priority_low);
    if (err != cudaSuccess) return err;
    
    // Create events for timing
    err = cudaEventCreate(&start_event_);
    if (err != cudaSuccess) return err;
    
    err = cudaEventCreate(&stop_event_);
    if (err != cudaSuccess) return err;
    
    // Allocate spectrum buffer (pinned memory for fast transfers)
    size_t spectrum_size = SPECTRUM_BINS * WATERFALL_HISTORY * sizeof(float);
    err = cudaMallocManaged(&d_spectrum_buffer_, spectrum_size);
    if (err != cudaSuccess) return err;
    
    // Advise GPU preferred location
    err = cudaMemAdvise(d_spectrum_buffer_, spectrum_size, 
                        cudaMemAdviseSetPreferredLocation, device_id);
    if (err != cudaSuccess) return err;
    
    // Allocate and initialize Q-table
    err = cudaMallocManaged(&d_qtable_, sizeof(QTableState));
    if (err != cudaSuccess) return err;
    
    // Initialize Q-table to small random values
    QTableState* h_qtable = new QTableState;
    for (uint32_t s = 0; s < NUM_STATES; ++s) {
        for (uint32_t a = 0; a < NUM_ACTIONS; ++a) {
            h_qtable->q_values[s][a] = 0.01f * ((float)rand() / RAND_MAX);
            h_qtable->eligibility_traces[s][a] = 0.0f;
        }
        h_qtable->visit_count[s] = 0;
    }
    h_qtable->current_state = 0;
    h_qtable->last_action = 0;
    h_qtable->total_reward = 0.0f;
    
    err = cudaMemcpy(d_qtable_, h_qtable, sizeof(QTableState), cudaMemcpyHostToDevice);
    delete h_qtable;
    if (err != cudaSuccess) return err;
    
    // Generate waveform bank
    err = generate_waveform_bank();
    if (err != cudaSuccess) return err;
    
    // Create FFT plan for spectrum analysis
    cufftResult fft_result = cufftPlan1d(&fft_plan_, SPECTRUM_BINS, CUFFT_R2C, 1);
    if (fft_result != CUFFT_SUCCESS) return cudaErrorUnknown;
    
    // Set FFT stream
    fft_result = cufftSetStream(fft_plan_, compute_stream_);
    if (fft_result != CUFFT_SUCCESS) return cudaErrorUnknown;
    
    return cudaSuccess;
}

cudaError_t AdaptiveJammingModule::process_spectrum(
    const float* d_spectrum_waterfall,
    ThreatSignature* d_threats,
    uint32_t num_threats,
    JammingParams* d_jamming_params,
    uint64_t timestamp_ns
) {
    cudaError_t err;
    
    // Record start time
    err = cudaEventRecord(start_event_, compute_stream_);
    if (err != cudaSuccess) return err;
    
    // Ensure we meet latency requirements
    if (num_threats > MAX_THREATS) {
        num_threats = MAX_THREATS;  // Process highest priority threats
    }
    
    // Configure kernel launch parameters
    const uint32_t block_size = 256;
    const uint32_t grid_size = (num_threats + block_size - 1) / block_size;
    
    // Launch adaptive jamming kernel
    adaptive_jamming_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
        d_spectrum_waterfall,
        d_threats,
        d_jamming_params,
        d_qtable_,
        num_threats,
        timestamp_ns
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Record stop time
    err = cudaEventRecord(stop_event_, compute_stream_);
    if (err != cudaSuccess) return err;
    
    // Wait for completion and measure time
    err = cudaEventSynchronize(stop_event_);
    if (err != cudaSuccess) return err;
    
    float elapsed_ms;
    err = cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    if (err != cudaSuccess) return err;
    
    // Update metrics
    float elapsed_us = elapsed_ms * 1000.0f;
    if (elapsed_us > MAX_LATENCY_US) {
        metrics_.deadline_misses++;
    }
    
    // Update running average of response time
    metrics_.average_response_time_us = 
        0.95f * metrics_.average_response_time_us + 0.05f * elapsed_us;
    
    metrics_.threats_detected += num_threats;
    metrics_.jamming_activated += num_threats;  // One jamming response per threat
    
    return cudaSuccess;
}

cudaError_t AdaptiveJammingModule::update_qlearning(float reward) {
    cudaError_t err;
    
    // Get current state from threats (simplified for now)
    uint32_t new_state = 0;  // Would be computed from current spectrum
    
    // Update Q-table
    const uint32_t block_size = 256;
    const uint32_t grid_size = (NUM_STATES + block_size - 1) / block_size;
    
    update_qtable_kernel<<<grid_size, block_size, 0, compute_stream_>>>(
        d_qtable_,
        reward,
        new_state
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Update effectiveness metric
    metrics_.jamming_effectiveness = 
        0.95f * metrics_.jamming_effectiveness + 0.05f * reward;
    
    return cudaSuccess;
}

cudaError_t AdaptiveJammingModule::generate_waveform_bank() {
    cudaError_t err;
    
    // Allocate waveform bank
    const uint32_t samples_per_waveform = 4096;
    const size_t bank_size = NUM_ACTIONS * samples_per_waveform * sizeof(float);
    
    err = cudaMalloc(&d_waveform_bank_, bank_size);
    if (err != cudaSuccess) return err;
    
    // Generate basic waveforms on CPU and transfer
    float* h_waveforms = new float[NUM_ACTIONS * samples_per_waveform];
    
    for (uint32_t w = 0; w < NUM_ACTIONS; ++w) {
        for (uint32_t i = 0; i < samples_per_waveform; ++i) {
            float t = (float)i / samples_per_waveform;
            float phase = 2.0f * 3.14159f * t;
            
            // Different base waveforms for each action
            switch (w % 4) {
                case 0:  // Sine wave
                    h_waveforms[w * samples_per_waveform + i] = sinf(phase * (w + 1));
                    break;
                case 1:  // Square wave
                    h_waveforms[w * samples_per_waveform + i] = 
                        (sinf(phase * (w + 1)) > 0) ? 1.0f : -1.0f;
                    break;
                case 2:  // Sawtooth
                    h_waveforms[w * samples_per_waveform + i] = 
                        2.0f * (t * (w + 1) - floorf(t * (w + 1))) - 1.0f;
                    break;
                case 3:  // Noise
                    h_waveforms[w * samples_per_waveform + i] = 
                        2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                    break;
            }
        }
    }
    
    err = cudaMemcpy(d_waveform_bank_, h_waveforms, bank_size, cudaMemcpyHostToDevice);
    delete[] h_waveforms;
    
    return err;
}

} // namespace ares::cew