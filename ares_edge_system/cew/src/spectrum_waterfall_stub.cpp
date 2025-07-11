#include "../include/spectrum_waterfall.h"
#include <cuda_runtime.h>
#include <cstring>

namespace ares {
namespace cew {

SpectrumWaterfall::SpectrumWaterfall() : history_depth_(1024), waterfall_index_(0), window_type_(WindowType::BLACKMAN) {
}

SpectrumWaterfall::~SpectrumWaterfall() {}

cudaError_t SpectrumWaterfall::initialize(int device_id, uint32_t history_depth, WindowType window) {
    history_depth_ = history_depth;
    window_type_ = window;
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::process_samples(const float2* samples, uint32_t num_samples, uint64_t timestamp_ns) {
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::detect_signals(DetectedSignal* signals, uint32_t* num_signals, uint32_t max_signals) {
    *num_signals = 0;
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::compute_statistics(SpectrumStats* stats) {
    if (stats) {
        stats->noise_floor_dbm = -100.0f;
        stats->peak_power_dbm = 0.0f;
        stats->avg_power_dbm = -50.0f;
        stats->active_bins = 0;
        stats->detected_signals = 0;
    }
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::update_noise_floor() {
    return cudaSuccess;
}

cudaError_t SpectrumWaterfall::generate_window_function() {
    return cudaSuccess;
}

} // namespace cew
} // namespace ares
