/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file defines lightweight struct versions to fix CUDA parameter size limitations
 */

#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>

namespace ares::cyber_em {

// Forward declarations of original structs
struct AttackVector;
struct SideChannelMeasurement;

// Lightweight version for kernel parameters
struct AttackVectorCompact {
    uint32_t attack_type;
    uint32_t target_id;
    float injection_freq_hz;
    float injection_power_dbm;
    float pulse_width_ns;
    float repetition_rate_hz;
    float success_probability;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    bool active;
    
    // Convert from full struct to compact
    static AttackVectorCompact FromFull(const AttackVector& full);
    
    // Convert compact struct to full version
    void ToFull(AttackVector& full) const;
};

// Lightweight version for kernel parameters
struct SideChannelMeasurementCompact {
    float information_leakage_bits;
    uint32_t key_bits_recovered;
    float confidence;
    
    // Convert from full struct to compact
    static SideChannelMeasurementCompact FromFull(const SideChannelMeasurement& full);
    
    // Convert compact struct to full version
    void ToFull(SideChannelMeasurement& full) const;
};

} // namespace ares::cyber_em
