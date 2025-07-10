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
 * @file spectrum_waterfall.cpp
 * @brief Implementation of real-time spectrum waterfall analysis
 */

#include "../include/spectrum_waterfall.h"
#include <cuda_runtime.h>
#include <cstring>

namespace ares::cew {

// Forward declare kernel wrapper
__global__ void generate_window_kernel(float* window, uint32_t size, WindowType type);

SpectrumWaterfall::SpectrumWaterfall()
    : d_iq_buffer_(nullptr)
    , d_window_function_(nullptr)
    , d_fft_buffer_(nullptr)
    , d_spectrum_mag_(nullptr)
    , d_spectrum_db_(nullptr)
    , d_waterfall_(nullptr)
    , d_noise_floor_(nullptr)
    , fft_plan_(0)
    , fft_stream_(nullptr)
    , detect_stream_(nullptr)
    , history_depth_(0)
    , waterfall_index_(0)
    , window_type_(WindowType::BLACKMAN) {
}

SpectrumWaterfall::~SpectrumWaterfall() {
    if (d_iq_buffer_) cudaFree(d_iq_buffer_);
    if (d_window_function_) cudaFree(d_window_function_);
    if (d_fft_buffer_) cudaFree(d_fft_buffer_);
    if (d_spectrum_mag_) cudaFree(d_spectrum_mag_);
    if (d_spectrum_db_) cudaFree(d_spectrum_db_);
    if (d_waterfall_) cudaFree(d_waterfall_);
    if (d_noise_floor_) cudaFree(d_noise_floor_);
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (fft_stream_) cudaStreamDestroy(fft_stream_);
    if (detect_stream_) cudaStreamDestroy(detect_stream_);
}

cudaError_t SpectrumWaterfall::initialize(
    int device_id,
    uint32_t history_depth,
    WindowType window
) {
    cudaError_t err;
    
    // Set device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return err;
    
    // Store configuration
    history_depth_ = history_depth;
    window_type_ = window;
    waterfall_index_ = 0;
    
    // Create streams with priorities
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    
    err = cudaStreamCreateWithPriority(&fft_stream_, cudaStreamNonBlocking, priority_high);
    if (err != cudaSuccess) return err;
    
    err = cudaStreamCreateWithPriority(&detect_stream_, cudaStreamNonBlocking, priority_low);
    if (err != cudaSuccess) return err;
    
    // Allocate IQ buffer (circular buffer for continuous processing)
    size_t iq_buffer_size = FFT_SIZE * 4 * sizeof(float2);  // 4x for buffering
    err = cudaMallocManaged(&d_iq_buffer_, iq_buffer_size);
    if (err != cudaSuccess) return err;
    
    // Allocate window function
    err = cudaMalloc(&d_window_function_, FFT_SIZE * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // Generate window function
    err = generate_window_function();
    if (err != cudaSuccess) return err;
    
    // Allocate FFT buffers
    err = cudaMalloc(&d_fft_buffer_, FFT_SIZE * sizeof(cufftComplex));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_spectrum_mag_, FFT_SIZE * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_spectrum_db_, FFT_SIZE * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // Allocate waterfall (2D array)
    size_t waterfall_size = history_depth_ * FFT_SIZE * sizeof(float);
    err = cudaMalloc(&d_waterfall_, waterfall_size);
    if (err != cudaSuccess) return err;
    
    // Initialize waterfall to noise floor
    err = cudaMemset(d_waterfall_, 0, waterfall_size);
    if (err != cudaSuccess) return err;
    
    // Allocate noise floor estimate
    err = cudaMalloc(&d_noise_floor_, FFT_SIZE * sizeof(float));
    if (err != cudaSuccess) return err;
    
    // Initialize noise floor
    float initial_noise = NOISE_FLOOR_DBM;
    for (uint32_t i = 0; i < FFT_SIZE; ++i) {
        err = cudaMemcpy(d_noise_floor_ + i, &initial_noise, sizeof(float), 
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
    }
    
    // Create FFT plan
    cufftResult fft_result = cufftPlan1d(&fft_plan_, FFT_SIZE, CUFFT_C2C, 1);
    if (fft_result != CUFFT_SUCCESS) return cudaErrorUnknown;
    
    // Associate FFT with stream
    fft_result = cufftSetStream(fft_plan_, fft_stream_);
    if (fft_result != CUFFT_SUCCESS) return cudaErrorUnknown;
    
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::process_samples(
    const float2* d_iq_samples,
    uint32_t num_samples,
    uint64_t timestamp_ns
) {
    cudaError_t err;
    
    // Process in overlapping windows
    uint32_t windows_to_process = (num_samples - FFT_SIZE) / WINDOW_STRIDE + 1;
    
    for (uint32_t w = 0; w < windows_to_process; ++w) {
        const float2* window_start = d_iq_samples + w * WINDOW_STRIDE;
        
        // Apply window function
        const uint32_t block_size = 256;
        const uint32_t grid_size = (FFT_SIZE + block_size - 1) / block_size;
        
        apply_window_kernel<<<grid_size, block_size, 0, fft_stream_>>>(
            window_start,
            d_window_function_,
            d_fft_buffer_,
            FFT_SIZE
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        // Execute FFT
        cufftResult fft_result = cufftExecC2C(
            fft_plan_,
            d_fft_buffer_,
            d_fft_buffer_,
            CUFFT_FORWARD
        );
        if (fft_result != CUFFT_SUCCESS) return cudaErrorUnknown;
        
        // Compute magnitude spectrum and convert to dB
        compute_magnitude_spectrum_kernel<<<grid_size, block_size, 0, fft_stream_>>>(
            d_fft_buffer_,
            d_spectrum_mag_,
            d_spectrum_db_,
            FFT_SIZE
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        // Update waterfall
        update_waterfall_kernel<<<grid_size, block_size, 0, fft_stream_>>>(
            d_spectrum_db_,
            d_waterfall_,
            waterfall_index_,
            FFT_SIZE,
            history_depth_
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        // Increment waterfall index
        waterfall_index_++;
        
        // Periodically update noise floor (every 32 windows)
        if ((waterfall_index_ % 32) == 0) {
            err = update_noise_floor();
            if (err != cudaSuccess) return err;
        }
    }
    
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::detect_signals(
    DetectedSignal* d_signals,
    uint32_t* num_signals,
    uint32_t max_signals
) {
    cudaError_t err;
    
    // Reset signal count
    err = cudaMemsetAsync(num_signals, 0, sizeof(uint32_t), detect_stream_);
    if (err != cudaSuccess) return err;
    
    // Launch detection kernel
    const uint32_t block_size = 256;
    const uint32_t grid_size = (FFT_SIZE + block_size - 1) / block_size;
    
    detect_signals_kernel<<<grid_size, block_size, 0, detect_stream_>>>(
        d_spectrum_db_,
        d_noise_floor_,
        d_signals,
        num_signals,
        DETECTION_THRESHOLD_DB,
        MIN_SIGNAL_BINS,
        FFT_SIZE
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Synchronize to ensure detection is complete
    err = cudaStreamSynchronize(detect_stream_);
    
    return err;
}

cudaError_t SpectrumWaterfall::compute_statistics(SpectrumStats* stats) {
    cudaError_t err;
    
    // Use CUB for parallel reduction to compute statistics
    size_t temp_storage_bytes = 0;
    
    // Get temporary storage requirements
    cub::DeviceReduce::Min(nullptr, temp_storage_bytes, d_spectrum_db_, 
                           &stats->noise_floor_dbm, FFT_SIZE);
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes * 3);  // For min, max, sum
    if (err != cudaSuccess) return err;
    
    // Compute min (noise floor approximation)
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_spectrum_db_, 
                           &stats->noise_floor_dbm, FFT_SIZE, detect_stream_);
    
    // Compute max (peak power)
    cub::DeviceReduce::Max((char*)d_temp_storage + temp_storage_bytes, 
                           temp_storage_bytes, d_spectrum_db_, 
                           &stats->peak_power_dbm, FFT_SIZE, detect_stream_);
    
    // Compute sum for average
    float sum = 0.0f;
    cub::DeviceReduce::Sum((char*)d_temp_storage + 2 * temp_storage_bytes, 
                           temp_storage_bytes, d_spectrum_db_, 
                           &sum, FFT_SIZE, detect_stream_);
    
    err = cudaStreamSynchronize(detect_stream_);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return err;
    }
    
    stats->avg_power_dbm = sum / FFT_SIZE;
    
    // Count active bins (simplified - would need custom kernel for accuracy)
    stats->active_bins = 0;
    stats->detected_signals = 0;
    
    cudaFree(d_temp_storage);
    
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::update_noise_floor() {
    // Launch noise floor estimation kernel
    const uint32_t block_size = 128;
    const uint32_t shared_mem_size = block_size * 3 * sizeof(float);
    
    estimate_noise_floor_kernel<<<FFT_SIZE, block_size, shared_mem_size, detect_stream_>>>(
        d_waterfall_,
        d_noise_floor_,
        FFT_SIZE,
        history_depth_
    );
    
    return cudaGetLastError();
}

cudaError_t SpectrumWaterfall::generate_window_function() {
    const uint32_t block_size = 256;
    const uint32_t grid_size = (FFT_SIZE + block_size - 1) / block_size;
    
    generate_window_kernel<<<grid_size, block_size>>>(
        d_window_function_,
        FFT_SIZE,
        window_type_
    );
    
    return cudaGetLastError();
}

} // namespace ares::cew