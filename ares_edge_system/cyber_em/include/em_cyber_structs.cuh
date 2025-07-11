/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file defines the EM cyber structures for the ARES Edge System
 */

#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <bitset>

namespace ares::cyber_em {

// Constants
constexpr uint32_t PROTOCOL_TEMPLATE_SIZE = 256;
constexpr uint32_t SIDE_CHANNEL_BANDS = 64;

// EM attack types
enum class EMAttackType : uint32_t {
    JAMMING = 0,
    SIGNAL_INJECTION = 1,
    TIMING_GLITCH = 2,
    SIDE_CHANNEL = 3,
    PROTOCOL_EXPLOIT = 4,
    POWER_ANALYSIS = 5,
    DEAUTH_ATTACK = 6,
    REPLAY_ATTACK = 7,
    GLITCH_INJECTION = 8,
    SIDE_CHANNEL_ANALYSIS = 9,
    PROTOCOL_FUZZING = 10,
    TEMPEST_INTERCEPT = 11,
    INJECTION_FAULT = 12
};

// EM defense modes
enum class EMDefenseMode : uint32_t {
    PASSIVE = 0,
    ACTIVE_JAMMING = 1,
    FREQUENCY_HOPPING = 2,
    SIGNAL_MASKING = 3,
    PROTOCOL_HARDENING = 4,
    DECEPTION = 5,
    FULL_SPECTRUM_SHIELD = 6,
    ACTIVE_SHIELDING = 7,
    NOISE_INJECTION = 8,
    DECEPTION_SIGNALS = 9
};

// Additional EM attack types for mapping
enum class AdditionalEMAttackType : uint32_t {
    GLITCH_INJECTION = 0,
    SIDE_CHANNEL_ANALYSIS = 1,
    PROTOCOL_FUZZING = 2,
    TEMPEST_INTERCEPT = 3,
    INJECTION_FAULT = 4,
    JAMMING_SELECTIVE = 5,
    MAN_IN_THE_MIDDLE = 6
};

// Additional EM defense modes for mapping
enum class AdditionalEMDefenseMode : uint32_t {
    ACTIVE_SHIELDING = 0,
    NOISE_INJECTION = 1,
    DECEPTION_SIGNALS = 2
};

// EM target
struct EMTarget {
    uint32_t target_id;
    float3 position;
    float frequency_hz;
    float bandwidth_hz;
    float power_level_dbm;
    uint8_t protocol_type;
    std::array<float, 128> em_signature;
    std::array<uint8_t, 64> protocol_pattern;
    bool is_vulnerable;
    float vulnerability_score;
    uint64_t last_seen_ns;
};

// Attack vector
struct AttackVector {
    EMAttackType attack_type;
    uint32_t target_id;
    float injection_freq_hz;
    float injection_power_dbm;
    float pulse_width_ns;
    float repetition_rate_hz;
    std::array<float, PROTOCOL_TEMPLATE_SIZE> waveform_template;
    float success_probability;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    bool active;
};

// Side channel measurement
struct SideChannelMeasurement {
    float frequency_bands[SIDE_CHANNEL_BANDS];
    float power_spectral_density[SIDE_CHANNEL_BANDS];
    float temporal_correlation[128]; // Reduced from 1024
    float information_leakage_bits;
    uint32_t key_bits_recovered;
    float confidence;
};

// Protocol vulnerability
struct ProtocolVulnerability {
    uint8_t protocol_id;
    std::string vulnerability_type;
    float exploit_difficulty;
    std::vector<uint8_t> exploit_sequence;
    float detection_probability;
    float impact_severity;
};

// Defense state
struct DefenseState {
    EMDefenseMode mode;
    float shield_frequencies[32];
    float shield_strengths[32];
    float noise_level_dbm;
    std::bitset<256> hardened_protocols;
    float emanation_reduction_db;
    bool deception_active;
};

} // namespace ares::cyber_em
