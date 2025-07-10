/**
 * @file threat_classifier_cnn.h
 * @brief Deep CNN for real-time modulation recognition and threat classification
 * 
 * Achieves <10ms inference for signal classification using optimized CNN
 * with TensorRT integration for maximum performance
 */

#ifndef ARES_CEW_THREAT_CLASSIFIER_CNN_H
#define ARES_CEW_THREAT_CLASSIFIER_CNN_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <memory>

namespace ares::cew {

// CNN Architecture Constants
constexpr uint32_t INPUT_SPECTRUM_SIZE = 1024;  // Reduced from 4096 for speed
constexpr uint32_t INPUT_TIME_STEPS = 64;       // Time-frequency representation
constexpr uint32_t NUM_CLASSES = 32;            // Threat types
constexpr float INFERENCE_TARGET_MS = 10.0f;    // 10ms target

// Threat Categories
enum class ThreatType : uint8_t {
    UNKNOWN = 0,
    RADAR_PULSE = 1,
    RADAR_CW = 2,
    RADAR_FMCW = 3,
    COMM_NARROWBAND = 4,
    COMM_WIDEBAND = 5,
    COMM_FREQUENCY_HOPPING = 6,
    JAMMING_BARRAGE = 7,
    JAMMING_SPOT = 8,
    JAMMING_SWEEP = 9,
    DATALINK_ENCRYPTED = 10,
    DATALINK_CLEAR = 11,
    GPS_L1 = 12,
    GPS_L2 = 13,
    DRONE_CONTROL = 14,
    DRONE_VIDEO = 15,
    CELLULAR_LTE = 16,
    CELLULAR_5G = 17,
    WIFI_24 = 18,
    WIFI_5 = 19,
    BLUETOOTH = 20,
    SATELLITE_UPLINK = 21,
    SATELLITE_DOWNLINK = 22,
    EMERGENCY_BEACON = 23,
    IFF_MODE_S = 24,
    ADS_B = 25,
    TACAN = 26,
    DME = 27,
    WEATHER_RADAR = 28,
    SYNTHETIC_AI = 29,
    QUANTUM_COMM = 30,
    RESERVED = 31
};

// CNN Layer Configuration
struct CNNArchitecture {
    // Optimized architecture for sub-10ms inference
    // Using depthwise separable convolutions and quantization
    
    // Layer 1: Initial feature extraction
    static constexpr uint32_t CONV1_FILTERS = 32;
    static constexpr uint32_t CONV1_SIZE = 7;
    static constexpr uint32_t CONV1_STRIDE = 2;
    
    // Layer 2-3: Depthwise separable blocks
    static constexpr uint32_t DEPTH2_MULTIPLIER = 1;
    static constexpr uint32_t POINT2_FILTERS = 64;
    
    // Layer 4-5: Reduced resolution processing
    static constexpr uint32_t DEPTH3_MULTIPLIER = 1;
    static constexpr uint32_t POINT3_FILTERS = 128;
    
    // Final layers
    static constexpr uint32_t FC1_UNITS = 256;
    static constexpr uint32_t FC2_UNITS = NUM_CLASSES;
};

// Classification Result
struct ClassificationResult {
    ThreatType threat_type;
    float confidence;
    float inference_time_ms;
    uint8_t modulation_params[16];  // Additional signal parameters
};

class ThreatClassifierCNN {
public:
    ThreatClassifierCNN();
    ~ThreatClassifierCNN();
    
    // Initialize CNN with pre-trained weights
    cudaError_t initialize(
        int device_id,
        const char* weights_file = nullptr,
        bool use_tensorrt = true
    );
    
    // Classify spectrum data
    cudaError_t classify(
        const float* d_spectrum_input,   // Time-frequency representation
        ClassificationResult* results,
        uint32_t batch_size = 1
    );
    
    // Update model with new training data (online learning)
    cudaError_t update_model(
        const float* d_training_data,
        const uint8_t* d_labels,
        uint32_t num_samples,
        float learning_rate = 0.001f
    );
    
    // Get model performance metrics
    float get_average_inference_time() const { return avg_inference_time_ms_; }
    float get_accuracy() const { return accuracy_; }
    
private:
    // cuDNN handles
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
    
    // Network descriptors
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t conv1_output_desc_;
    cudnnTensorDescriptor_t depth2_output_desc_;
    cudnnTensorDescriptor_t point2_output_desc_;
    cudnnTensorDescriptor_t fc_input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    
    // Filter descriptors
    cudnnFilterDescriptor_t conv1_filter_desc_;
    cudnnFilterDescriptor_t depth2_filter_desc_;
    cudnnFilterDescriptor_t point2_filter_desc_;
    
    // Convolution descriptors
    cudnnConvolutionDescriptor_t conv1_desc_;
    cudnnConvolutionDescriptor_t depth2_desc_;
    cudnnConvolutionDescriptor_t point2_desc_;
    
    // Activation descriptor
    cudnnActivationDescriptor_t relu_desc_;
    
    // Device memory for weights and activations
    float* d_conv1_weights_;
    float* d_conv1_bias_;
    float* d_depth2_weights_;
    float* d_point2_weights_;
    float* d_point2_bias_;
    float* d_fc1_weights_;
    float* d_fc1_bias_;
    float* d_fc2_weights_;
    float* d_fc2_bias_;
    
    // Workspace
    void* d_workspace_;
    size_t workspace_size_;
    
    // Activation buffers
    float* d_conv1_output_;
    float* d_depth2_output_;
    float* d_point2_output_;
    float* d_fc1_output_;
    float* d_output_;
    
    // Performance tracking
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    float avg_inference_time_ms_;
    float accuracy_;
    
    // Internal methods
    cudaError_t create_network_descriptors();
    cudaError_t allocate_network_memory();
    cudaError_t load_pretrained_weights(const char* weights_file);
    cudaError_t forward_pass(const float* input, float* output, uint32_t batch_size);
};

// Optimized CUDA kernels for custom layers
__global__ void preprocess_spectrum_kernel(
    const float* __restrict__ raw_spectrum,
    float* __restrict__ preprocessed,
    uint32_t spectrum_size,
    uint32_t time_steps
);

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ filters,
    float* __restrict__ output,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t filter_size,
    uint32_t stride
);

__global__ void quantized_linear_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    int32_t* __restrict__ output,
    uint32_t input_size,
    uint32_t output_size,
    float scale_factor
);

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t num_classes,
    uint32_t batch_size
);

} // namespace ares::cew

#endif // ARES_CEW_THREAT_CLASSIFIER_CNN_H