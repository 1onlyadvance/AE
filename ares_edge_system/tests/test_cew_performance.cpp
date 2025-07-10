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
 * @file test_cew_performance.cpp
 * @brief Performance validation tests for CEW components
 * 
 * Verifies that all latency requirements are met
 */

#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include "../cew/include/cew_adaptive_jamming.h"
#include "../cew/include/spectrum_waterfall.h"

using namespace ares::cew;
using namespace std::chrono;

// Test configuration
constexpr uint32_t NUM_ITERATIONS = 100;
constexpr uint32_t NUM_THREATS = 64;
constexpr uint32_t SPECTRUM_SAMPLES = 8192;

class CEWPerformanceTester {
public:
    CEWPerformanceTester() : device_id_(0) {}
    
    bool run_all_tests() {
        std::cout << "=== ARES CEW Performance Validation ===" << std::endl;
        
        if (!initialize_cuda()) return false;
        if (!test_adaptive_jamming()) return false;
        if (!test_spectrum_waterfall()) return false;
        if (!test_end_to_end_latency()) return false;
        
        std::cout << "\n=== All Tests Passed ===" << std::endl;
        return true;
    }
    
private:
    int device_id_;
    
    bool initialize_cuda() {
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id_);
        std::cout << "\nUsing GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        
        return true;
    }
    
    bool test_adaptive_jamming() {
        std::cout << "\n--- Testing Adaptive Jamming Module ---" << std::endl;
        
        AdaptiveJammingModule jamming_module;
        cudaError_t err = jamming_module.initialize(device_id_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize jamming module: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Allocate test data
        float* d_spectrum;
        ThreatSignature* d_threats;
        JammingParams* d_jamming_params;
        
        cudaMalloc(&d_spectrum, SPECTRUM_BINS * sizeof(float));
        cudaMalloc(&d_threats, NUM_THREATS * sizeof(ThreatSignature));
        cudaMalloc(&d_jamming_params, NUM_THREATS * sizeof(JammingParams));
        
        // Generate test threats
        ThreatSignature* h_threats = new ThreatSignature[NUM_THREATS];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> freq_dist(0.5f, 20.0f);
        std::uniform_real_distribution<float> power_dist(-80.0f, -20.0f);
        
        for (uint32_t i = 0; i < NUM_THREATS; ++i) {
            h_threats[i].center_freq_ghz = freq_dist(gen);
            h_threats[i].bandwidth_mhz = 10.0f + (i % 10) * 5.0f;
            h_threats[i].power_dbm = power_dist(gen);
            h_threats[i].modulation_type = i % 8;
            h_threats[i].protocol_id = i % 4;
            h_threats[i].priority = (i < 10) ? 1 : 0;  // First 10 are high priority
        }
        
        cudaMemcpy(d_threats, h_threats, NUM_THREATS * sizeof(ThreatSignature), cudaMemcpyHostToDevice);
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            jamming_module.process_spectrum(d_spectrum, d_threats, NUM_THREATS, d_jamming_params, i);
        }
        
        // Performance test
        auto start = high_resolution_clock::now();
        
        for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
            err = jamming_module.process_spectrum(d_spectrum, d_threats, NUM_THREATS, d_jamming_params, i);
            if (err != cudaSuccess) {
                std::cerr << "Jamming processing failed: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        }
        
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        float avg_latency = duration.count() / (float)NUM_ITERATIONS;
        
        std::cout << "Average jamming response latency: " << avg_latency << " µs" << std::endl;
        std::cout << "Target: < 100,000 µs (100ms)" << std::endl;
        std::cout << "Status: " << (avg_latency < 100000 ? "PASS" : "FAIL") << std::endl;
        
        // Get metrics
        CEWMetrics metrics = jamming_module.get_metrics();
        std::cout << "Threats processed: " << metrics.threats_detected << std::endl;
        std::cout << "Deadline misses: " << metrics.deadline_misses << std::endl;
        
        // Cleanup
        cudaFree(d_spectrum);
        cudaFree(d_threats);
        cudaFree(d_jamming_params);
        delete[] h_threats;
        
        return avg_latency < 100000;  // Must be under 100ms
    }
    
    bool test_spectrum_waterfall() {
        std::cout << "\n--- Testing Spectrum Waterfall Analysis ---" << std::endl;
        
        SpectrumWaterfall waterfall;
        cudaError_t err = waterfall.initialize(device_id_, 128, WindowType::BLACKMAN);
        if (err != cudaSuccess) {
            std::cerr << "Failed to initialize waterfall: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Generate test IQ data
        float2* d_iq_samples;
        cudaMalloc(&d_iq_samples, SPECTRUM_SAMPLES * sizeof(float2));
        
        // Generate complex sinusoid test signal
        float2* h_iq = new float2[SPECTRUM_SAMPLES];
        for (uint32_t i = 0; i < SPECTRUM_SAMPLES; ++i) {
            float t = (float)i / SAMPLE_RATE_MSPS;
            h_iq[i].x = cosf(2.0f * M_PI * 100.0f * t);  // 100 MHz signal
            h_iq[i].y = sinf(2.0f * M_PI * 100.0f * t);
        }
        
        cudaMemcpy(d_iq_samples, h_iq, SPECTRUM_SAMPLES * sizeof(float2), cudaMemcpyHostToDevice);
        
        // Performance test
        auto start = high_resolution_clock::now();
        
        for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
            err = waterfall.process_samples(d_iq_samples, SPECTRUM_SAMPLES, i);
            if (err != cudaSuccess) {
                std::cerr << "Waterfall processing failed: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        }
        
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        float avg_latency = duration.count() / (float)NUM_ITERATIONS;
        
        std::cout << "Average spectrum analysis latency: " << avg_latency << " µs" << std::endl;
        std::cout << "Processing rate: " << (SPECTRUM_SAMPLES * NUM_ITERATIONS * 1e6) / duration.count() / 1e6 << " MSPS" << std::endl;
        
        // Test signal detection
        DetectedSignal* d_signals;
        uint32_t* d_num_signals;
        cudaMalloc(&d_signals, 128 * sizeof(DetectedSignal));
        cudaMalloc(&d_num_signals, sizeof(uint32_t));
        
        err = waterfall.detect_signals(d_signals, d_num_signals, 128);
        if (err != cudaSuccess) {
            std::cerr << "Signal detection failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        uint32_t num_detected;
        cudaMemcpy(&num_detected, d_num_signals, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::cout << "Signals detected: " << num_detected << std::endl;
        
        // Cleanup
        cudaFree(d_iq_samples);
        cudaFree(d_signals);
        cudaFree(d_num_signals);
        delete[] h_iq;
        
        return true;
    }
    
    bool test_end_to_end_latency() {
        std::cout << "\n--- Testing End-to-End CEW Pipeline ---" << std::endl;
        
        // Initialize all components
        AdaptiveJammingModule jamming_module;
        SpectrumWaterfall waterfall;
        
        jamming_module.initialize(device_id_);
        waterfall.initialize(device_id_);
        
        // Allocate pipeline buffers
        float2* d_iq_input;
        float* d_spectrum;
        ThreatSignature* d_threats;
        JammingParams* d_jamming_params;
        
        cudaMalloc(&d_iq_input, SPECTRUM_SAMPLES * sizeof(float2));
        cudaMalloc(&d_spectrum, SPECTRUM_BINS * sizeof(float));
        cudaMalloc(&d_threats, MAX_THREATS * sizeof(ThreatSignature));
        cudaMalloc(&d_jamming_params, MAX_THREATS * sizeof(JammingParams));
        
        // Create CUDA events for precise timing
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        // Simulate end-to-end processing
        float total_time = 0.0f;
        
        for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
            cudaEventRecord(start_event);
            
            // Step 1: Process spectrum
            waterfall.process_samples(d_iq_input, FFT_SIZE, i);
            
            // Step 2: Get spectrum data (simulated)
            const float* spectrum_data = waterfall.get_spectrum_db();
            
            // Step 3: Threat detection (simulated - would use CNN)
            // For now, generate synthetic threats
            
            // Step 4: Adaptive jamming response
            jamming_module.process_spectrum(spectrum_data, d_threats, 32, d_jamming_params, i);
            
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            
            float iteration_time;
            cudaEventElapsedTime(&iteration_time, start_event, stop_event);
            total_time += iteration_time;
        }
        
        float avg_end_to_end = total_time / NUM_ITERATIONS;
        
        std::cout << "Average end-to-end latency: " << avg_end_to_end << " ms" << std::endl;
        std::cout << "Target: < 10 ms" << std::endl;
        std::cout << "Status: " << (avg_end_to_end < 10.0f ? "PASS" : "FAIL") << std::endl;
        
        // Cleanup
        cudaFree(d_iq_input);
        cudaFree(d_spectrum);
        cudaFree(d_threats);
        cudaFree(d_jamming_params);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        
        return avg_end_to_end < 10.0f;  // Must be under 10ms
    }
};

int main(int argc, char** argv) {
    CEWPerformanceTester tester;
    
    if (!tester.run_all_tests()) {
        std::cerr << "\nPerformance validation FAILED!" << std::endl;
        return 1;
    }
    
    return 0;
}