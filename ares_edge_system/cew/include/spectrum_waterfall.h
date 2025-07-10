/**
 * @file spectrum_waterfall.h
 * @brief Real-time spectrum waterfall analysis for CEW
 * 
 * Provides continuous spectrum monitoring with sliding window FFT
 * and threat detection capabilities
 */

#ifndef ARES_CEW_SPECTRUM_WATERFALL_H
#define ARES_CEW_SPECTRUM_WATERFALL_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>

namespace ares::cew {

// Waterfall configuration
constexpr uint32_t FFT_SIZE = 4096;
constexpr uint32_t OVERLAP_FACTOR = 4;  // 75% overlap
constexpr uint32_t WINDOW_STRIDE = FFT_SIZE / OVERLAP_FACTOR;
constexpr uint32_t MAX_HISTORY_DEPTH = 256;
constexpr float SAMPLE_RATE_MSPS = 2000.0f;  // 2 GSPS

// Detection parameters
constexpr float NOISE_FLOOR_DBM = -110.0f;
constexpr float DETECTION_THRESHOLD_DB = 10.0f;  // 10dB above noise floor
constexpr uint32_t MIN_SIGNAL_BINS = 4;  // Minimum width for detection

// Window functions
enum class WindowType : uint8_t {
    RECTANGULAR = 0,
    HANNING = 1,
    HAMMING = 2,
    BLACKMAN = 3,
    KAISER = 4,
    FLAT_TOP = 5
};

// Spectrum statistics
struct SpectrumStats {
    float noise_floor_dbm;
    float peak_power_dbm;
    float avg_power_dbm;
    uint32_t active_bins;
    uint32_t detected_signals;
};

// Signal detection result
struct DetectedSignal {
    float center_freq_mhz;
    float bandwidth_mhz;
    float power_dbm;
    float snr_db;
    uint32_t start_bin;
    uint32_t end_bin;
    uint8_t confidence;  // 0-100%
    uint8_t padding[3];
};

class SpectrumWaterfall {
public:
    SpectrumWaterfall();
    ~SpectrumWaterfall();
    
    // Initialize with device and parameters
    cudaError_t initialize(
        int device_id,
        uint32_t history_depth = MAX_HISTORY_DEPTH,
        WindowType window = WindowType::BLACKMAN
    );
    
    // Process raw IQ samples
    cudaError_t process_samples(
        const float2* d_iq_samples,  // Complex IQ samples
        uint32_t num_samples,
        uint64_t timestamp_ns
    );
    
    // Get current spectrum and waterfall
    const float* get_spectrum_db() const { return d_spectrum_db_; }
    const float* get_waterfall() const { return d_waterfall_; }
    
    // Detect signals in current spectrum
    cudaError_t detect_signals(
        DetectedSignal* d_signals,
        uint32_t* num_signals,
        uint32_t max_signals
    );
    
    // Get spectrum statistics
    cudaError_t compute_statistics(SpectrumStats* stats);
    
    // Update noise floor estimate
    cudaError_t update_noise_floor();
    
private:
    // Device memory
    float2* d_iq_buffer_;           // Ring buffer for IQ samples
    float* d_window_function_;      // FFT window
    cufftComplex* d_fft_buffer_;    // FFT working buffer
    float* d_spectrum_mag_;         // Magnitude spectrum
    float* d_spectrum_db_;          // Spectrum in dB
    float* d_waterfall_;            // 2D waterfall history
    float* d_noise_floor_;          // Per-bin noise floor estimate
    
    // CUDA resources
    cufftHandle fft_plan_;
    cudaStream_t fft_stream_;
    cudaStream_t detect_stream_;
    
    // Configuration
    uint32_t history_depth_;
    uint32_t waterfall_index_;
    WindowType window_type_;
    
    // Internal methods
    cudaError_t generate_window_function();
    cudaError_t apply_window(const float2* input, cufftComplex* output);
};

// CUDA Kernels
__global__ void apply_window_kernel(
    const float2* __restrict__ iq_samples,
    const float* __restrict__ window,
    cufftComplex* __restrict__ windowed_output,
    uint32_t fft_size
);

__global__ void compute_magnitude_spectrum_kernel(
    const cufftComplex* __restrict__ fft_output,
    float* __restrict__ magnitude,
    float* __restrict__ spectrum_db,
    uint32_t spectrum_bins
);

__global__ void update_waterfall_kernel(
    const float* __restrict__ spectrum_db,
    float* __restrict__ waterfall,
    uint32_t waterfall_index,
    uint32_t spectrum_bins,
    uint32_t history_depth
);

__global__ void detect_signals_kernel(
    const float* __restrict__ spectrum_db,
    const float* __restrict__ noise_floor,
    DetectedSignal* __restrict__ signals,
    uint32_t* __restrict__ signal_count,
    float threshold_db,
    uint32_t min_bins,
    uint32_t spectrum_bins
);

__global__ void estimate_noise_floor_kernel(
    const float* __restrict__ waterfall,
    float* __restrict__ noise_floor,
    uint32_t spectrum_bins,
    uint32_t history_depth
);

} // namespace ares::cew

#endif // ARES_CEW_SPECTRUM_WATERFALL_H