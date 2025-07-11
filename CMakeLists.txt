# Set a modern CMake version
cmake_minimum_required(VERSION 3.18)
project(ARES_Edge_System LANGUAGES CXX CUDA)

# --- Project Standards and Options ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA compiler flags for relaxed constexpr
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda -DSKIP_CUDNN")

option(USE_STUBS "Build with stubbed-out implementations" ON)

# --- Find Required Packages ---
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

# --- Source File Lists ---
set(ARES_REAL_SOURCES
    "ares_edge_system/ares_cpu_demo.cpp"
    "ares_edge_system/ares_edge_system.cpp"
    "ares_edge_system/ares_edge_system_optimized.cpp"
    "ares_edge_system/ares_edge_system_optimized_original.cpp"
    "ares_edge_system/ares_edge_system_original.cpp"
    "ares_edge_system/ares_edge_system_simple.cpp"
    "ares_edge_system/backscatter/src/backscatter_communication_system.cpp"
    "ares_edge_system/backscatter/src/rf_energy_harvesting_system.cpp"
    "ares_edge_system/cew/kernels/adaptive_jamming_kernel.cu"
    "ares_edge_system/cew/kernels/adaptive_jamming_kernel_optimized.cu"
    "ares_edge_system/cew/kernels/spectrum_waterfall_kernel.cu"
    "ares_edge_system/cew/kernels/threat_classifier_kernel.cu"
    "ares_edge_system/core/quantum_core.cu"
    "ares_edge_system/core/quantum_resilient_core.cpp"
    "ares_edge_system/countermeasures/src/last_man_standing_coordinator.cu"
    "ares_edge_system/countermeasures/src/self_destruct_protocol.cu"
    "ares_edge_system/cyber_em/src/em_cyber_controller.cu"
    "ares_edge_system/cyber_em/src/protocol_exploitation_engine.cu"
    "ares_edge_system/digital_twin/kernels/physics_simulation_kernels.cu"
    "ares_edge_system/digital_twin/kernels/state_sync_kernels.cu"
    "ares_edge_system/digital_twin/src/predictive_simulation_engine.cu"
    "ares_edge_system/digital_twin/src/realtime_state_sync.cu"
    "ares_edge_system/federated_learning/src/distributed_slam_engine.cu"
    "ares_edge_system/federated_learning/src/federated_learning_coordinator.cu"
    "ares_edge_system/federated_learning/src/homomorphic_computation_engine.cu"
    "ares_edge_system/federated_learning/src/neuromorphic_processor_interface.cu"
    "ares_edge_system/federated_learning/src/secure_multiparty_computation.cu"
    "ares_edge_system/identity/src/hardware_attestation_system.cu"
    "ares_edge_system/identity/src/hot_swap_identity_manager.cu"
    "ares_edge_system/main.cpp"
    "ares_edge_system/main_optimized.cpp"
    "ares_edge_system/neuromorphic/kernels/neuromorphic_simulation_kernels.cu"
    "ares_edge_system/neuromorphic/kernels/spike_encoding_kernel.cu"
    "ares_edge_system/neuromorphic/src/loihi2_hardware_abstraction.cpp"
    "ares_edge_system/optical_stealth/src/dynamic_metamaterial_controller.cu"
    "ares_edge_system/optical_stealth/src/multi_spectral_fusion_engine.cu"
    "ares_edge_system/optical_stealth/src/rioss_synthesis_engine.cu"
    "ares_edge_system/orchestrator/chronopath_engine.cpp"
    "ares_edge_system/orchestrator/drpp_chronopath_engine.cu"
    "ares_edge_system/swarm/kernels/auction_optimization_kernels.cu"
    "ares_edge_system/swarm/src/byzantine_consensus_engine.cpp"
    "ares_edge_system/swarm/src/distributed_task_auction.cpp"
    "ares_edge_system/tests/integration_tests.cpp"
    "ares_edge_system/tests/performance_benchmarks.cu"
    "ares_edge_system/tests/test_cew_performance.cpp"
    "ares_edge_system/tests/test_chaos_engine.cpp"
    "ares_edge_system/unified_quantum_ares.cpp"
    "ares_edge_system/unreal/ARESEdgePlugin/Source/ARESEdgePlugin/Private/ARESGameMode.cpp"
    "ares_edge_system/unreal/ARESEdgePlugin/Source/ARESEdgePlugin/Private/ARESGameMode_Optimized.cpp"
)
set(ARES_STUB_SOURCES
    "ares_edge_system/cew/src/cew_adaptive_jamming_stub.cpp"
    "ares_edge_system/cew/src/spectrum_waterfall_stub.cpp"
    "ares_edge_system/countermeasures/src/chaos_induction_engine_stub.cpp"
)

# --- Logic to Select Final Sources ---
# Start with the list of real source files
set(ARES_FINAL_SOURCES ${ARES_REAL_SOURCES})

if(USE_STUBS)
  message(STATUS "Building with STUBBED implementations.")
  # Add the stubs to the final list
  list(APPEND ARES_FINAL_SOURCES ${ARES_STUB_SOURCES})
else()
  message(STATUS "Building with REAL implementations.")
  # When not using stubs, add the real counterparts
  list(APPEND ARES_FINAL_SOURCES
      "ares_edge_system/cew/src/cew_adaptive_jamming.cu"
      "ares_edge_system/cew/src/spectrum_waterfall.cu"
      "ares_edge_system/countermeasures/src/chaos_induction_engine.cu"
  )
endif()

# --- Create the Executable ---
add_executable(ares_edge_system ${ARES_FINAL_SOURCES})

# --- Include Directories and Library Linking ---
target_include_directories(ares_edge_system PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
)

target_link_libraries(ares_edge_system PRIVATE
    Threads::Threads
    ${CUDAToolkit_LIBRARIES}
    MPI::MPI_CXX
    crypto++
    nccl
)

# Add include directories for the compiler to find header files.
include_directories(
    /usr/local/cuda-12.3/targets/x86_64-linux/include
    /usr/include/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)