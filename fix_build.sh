#!/bin/bash

# Fix ARES Edge System Build Issues

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing ARES Edge System build issues...${NC}"

# Navigate to project root
cd /workspaces/AE

# Create missing directories
echo -e "${BLUE}Creating missing directories...${NC}"
mkdir -p ares_edge_system/neuromorphic/include
mkdir -p ares_edge_system/neuromorphic/src

# Create the Loihi2 stub header
echo -e "${BLUE}Creating Loihi2 spike encoding stub...${NC}"
cat > ares_edge_system/neuromorphic/include/loihi2_spike_encoding.h << 'INNEREOF'
#pragma once

// ARES Edge System - Loihi2 Spike Encoding Stub
// This is a placeholder for the neuromorphic spike encoding module

#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include <cmath>

namespace ares {
namespace neuromorphic {

constexpr float SPIKE_THRESHOLD = 1.0f;
constexpr float REFRACTORY_PERIOD = 2.0f;
constexpr float TIME_CONSTANT = 20.0f;

struct Loihi2NeuronParams {
    float threshold = SPIKE_THRESHOLD;
    float reset_potential = 0.0f;
    float leak_rate = 0.1f;
    float weight_scale = 1.0f;
    float bias = 0.0f;
    uint32_t refractory_cycles = 3;
};

struct SpikeTrain {
    std::vector<float> spike_times;
    uint32_t neuron_id;
    
    void add_spike(float time) {
        spike_times.push_back(time);
    }
    
    size_t count() const {
        return spike_times.size();
    }
};

class Loihi2SpikeEncoder {
private:
    Loihi2NeuronParams params_;
    std::vector<float> membrane_potentials_;
    std::vector<float> refractory_timers_;
    size_t num_neurons_;
    
public:
    explicit Loihi2SpikeEncoder(size_t num_neurons = 1024)
        : num_neurons_(num_neurons)
        , membrane_potentials_(num_neurons, 0.0f)
        , refractory_timers_(num_neurons, 0.0f) {}
    
    std::vector<SpikeTrain> encode_rate(const std::vector<float>& input_rates, 
                                        float duration_ms = 1000.0f) {
        std::vector<SpikeTrain> spike_trains(input_rates.size());
        
        for (size_t i = 0; i < input_rates.size(); ++i) {
            spike_trains[i].neuron_id = i;
            float rate = input_rates[i];
            
            if (rate > 0) {
                float inter_spike_interval = 1000.0f / rate;
                for (float t = 0; t < duration_ms; t += inter_spike_interval) {
                    spike_trains[i].add_spike(t);
                }
            }
        }
        
        return spike_trains;
    }
    
    std::vector<SpikeTrain> encode_temporal(const std::vector<float>& input_values,
                                           float max_delay_ms = 100.0f) {
        std::vector<SpikeTrain> spike_trains(input_values.size());
        
        for (size_t i = 0; i < input_values.size(); ++i) {
            spike_trains[i].neuron_id = i;
            float normalized = std::max(0.0f, std::min(1.0f, input_values[i]));
            float spike_time = max_delay_ms * (1.0f - normalized);
            
            if (normalized > 0.01f) {
                spike_trains[i].add_spike(spike_time);
            }
        }
        
        return spike_trains;
    }
    
    std::vector<bool> step_lif(const std::vector<float>& inputs, float dt = 1.0f) {
        std::vector<bool> spikes(num_neurons_, false);
        
        for (size_t i = 0; i < num_neurons_ && i < inputs.size(); ++i) {
            if (refractory_timers_[i] > 0) {
                refractory_timers_[i] -= dt;
                continue;
            }
            
            membrane_potentials_[i] += inputs[i] * params_.weight_scale + params_.bias;
            membrane_potentials_[i] *= (1.0f - params_.leak_rate * dt);
            
            if (membrane_potentials_[i] >= params_.threshold) {
                spikes[i] = true;
                membrane_potentials_[i] = params_.reset_potential;
                refractory_timers_[i] = params_.refractory_cycles * dt;
            }
        }
        
        return spikes;
    }
    
    void set_params(const Loihi2NeuronParams& params) {
        params_ = params;
    }
    
    void reset() {
        std::fill(membrane_potentials_.begin(), membrane_potentials_.end(), 0.0f);
        std::fill(refractory_timers_.begin(), refractory_timers_.end(), 0.0f);
    }
};

inline float compute_spike_rate(const SpikeTrain& train, float window_ms = 1000.0f) {
    return static_cast<float>(train.count()) / window_ms * 1000.0f;
}

inline float compute_isi_variance(const SpikeTrain& train) {
    if (train.spike_times.size() < 2) return 0.0f;
    
    std::vector<float> isis;
    for (size_t i = 1; i < train.spike_times.size(); ++i) {
        isis.push_back(train.spike_times[i] - train.spike_times[i-1]);
    }
    
    float mean = 0.0f;
    for (float isi : isis) mean += isi;
    mean /= isis.size();
    
    float variance = 0.0f;
    for (float isi : isis) {
        float diff = isi - mean;
        variance += diff * diff;
    }
    
    return variance / isis.size();
}

} // namespace neuromorphic
} // namespace ares
INNEREOF

# Check if we need to fix the cudnn.h issue
if grep -q "#include <cudnn.h>" ares_edge_system/backscatter/src/backscatter_communication_system.cpp 2>/dev/null; then
    echo -e "${BLUE}Fixing cuDNN include...${NC}"
    # Comment out the cudnn.h include
    sed -i 's|#include <cudnn.h>|// #include <cudnn.h> // Commented out - cuDNN not available|g' \
        ares_edge_system/backscatter/src/backscatter_communication_system.cpp
fi

# Now rebuild
echo -e "${BLUE}Rebuilding ARES Edge System...${NC}"
cd build
make clean
make -j4

echo -e "${GREEN}Build fix script completed!${NC}"
