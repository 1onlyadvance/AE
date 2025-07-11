/**
 * @file cew_adaptive_jamming.h
 * @brief Cognitive Electronic Warfare Adaptive Jamming Module
 * 
 * Achieves <100ms threat detection to jamming response using Q-learning
 * with eligibility traces for real-time adaptation.
 */

#ifndef ARES_CEW_ADAPTIVE_JAMMING_H
#define ARES_CEW_ADAPTIVE_JAMMING_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>
#include <array>

namespace ares::cew {

// CEW Configuration Constants
constexpr uint32_t SPECTRUM_BINS = 4096;
constexpr uint32_t WATERFALL_HISTORY = 256;
constexpr float FREQ_MIN_GHZ = 0.1f;
constexpr float FREQ_MAX_GHZ = 40.0f;
constexpr uint32_t MAX_THREATS = 128;
constexpr uint32_t MAX_LATENCY_US = 100000;  // 100ms hard deadline

// Q-Learning Parameters
constexpr float ALPHA = 0.1f;          // Learning rate
constexpr float GAMMA = 0.95f;         // Discount factor
constexpr float EPSILON = 0.05f;       // Exploration rate
constexpr uint32_t NUM_ACTIONS = 16;   // Jamming strategies
constexpr uint32_t NUM_STATES = 256;   // Quantized threat states

// Jamming Strategies
enum class JammingStrategy : uint8_t {
    BARRAGE_NARROW = 0,
    BARRAGE_WIDE = 1,
    SPOT_JAMMING = 2,
    SWEEP_SLOW = 3,
    SWEEP_FAST = 4,
    PULSE_JAMMING = 5,
    NOISE_MODULATED = 6,
    DECEPTIVE_REPEAT = 7,
    PROTOCOL_AWARE = 8,
    COGNITIVE_ADAPTIVE = 9,
    FREQUENCY_HOPPING = 10,
    TIME_SLICED = 11,
    POWER_CYCLING = 12,
    MIMO_SPATIAL = 13,
    PHASE_ALIGNED = 14,
    NULL_STEERING = 15
};

// Threat Classification
struct ThreatSignature {
    float center_freq_ghz;
    float bandwidth_mhz;
    float power_dbm;
    uint8_t modulation_type;
    uint8_t protocol_id;
    uint8_t priority;
    uint8_t padding[2];
};

// Jamming Parameters
struct JammingParams {
    float center_freq_ghz;
    float bandwidth_mhz;
    float power_watts;
    uint8_t strategy;
    uint8_t waveform_id;
    uint16_t duration_ms;
    float phase_offset;
    float sweep_rate_mhz_per_sec;
};

// Q-Table State
struct QTableState {
    float q_values[NUM_STATES][NUM_ACTIONS];
    float eligibility_traces[NUM_STATES][NUM_ACTIONS];
    uint32_t visit_count[NUM_STATES];
    uint32_t current_state;
    uint32_t last_action;
    float total_reward;
};

// Performance Metrics
struct CEWMetrics {
    uint64_t threats_detected;
    uint64_t jamming_activated;
    float average_response_time_us;
    float jamming_effectiveness;
    uint32_t deadline_misses;
};

class AdaptiveJammingModule {
public:
    AdaptiveJammingModule();
    ~AdaptiveJammingModule();
    
    // Initialize module with GPU device
    cudaError_t initialize(int device_id);
    
    // Process spectrum and generate jamming response
    cudaError_t process_spectrum(
        const float* d_spectrum_waterfall,
        ThreatSignature* d_threats,
        uint32_t num_threats,
        JammingParams* d_jamming_params,
        uint64_t timestamp_ns
    );
    
    // Update Q-learning model with reward feedback
    cudaError_t update_qlearning(float reward);
    
    // Get performance metrics
    CEWMetrics get_metrics() const { return metrics_; }
    
private:
    // Initialization state
    bool initialized_;
    
    // Device pointers
    float* d_spectrum_buffer_;
    QTableState* d_qtable_;
    float* d_waveform_bank_;
    cufftHandle fft_plan_;
    
    // CUDA resources
    cudaStream_t compute_stream_;
    cudaStream_t transfer_stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Metrics
    CEWMetrics metrics_;
    
    // Internal methods
    void cleanup();
    cudaError_t generate_waveform_bank();
    cudaError_t analyze_threats(const float* spectrum, ThreatSignature* threats);
};

// CUDA Kernels
__global__ void adaptive_jamming_kernel(
    const float* __restrict__ spectrum_waterfall,
    const ThreatSignature* __restrict__ threats,
    JammingParams* __restrict__ jamming_params,
    QTableState* q_state,
    uint32_t num_threats,
    uint64_t timestamp_ns
);

__global__ void update_qtable_kernel(
    QTableState* q_state,
    float reward,
    uint32_t new_state
);

__global__ void generate_jamming_waveform_kernel(
    float* __restrict__ waveform_out,
    const JammingParams* __restrict__ params,
    const float* __restrict__ waveform_bank,
    uint32_t samples_per_symbol
);

} // namespace ares::cew

#endif // ARES_CEW_ADAPTIVE_JAMMING_H