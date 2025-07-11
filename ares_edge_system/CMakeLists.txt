# CMake build script for the ARES Edge System

cmake_minimum_required(VERSION 3.18)

project(ARESEdgeSystem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling OpenMP support.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler \"-Wno-psabi\"")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator
    ${CMAKE_CURRENT_SOURCE_DIR}/optical_stealth/include
    ${CMAKE_CURRENT_SOURCE_DIR}/countermeasures/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cyber_em/include
    ${CMAKE_CURRENT_SOURCE_DIR}/federated_learning/include
    ${CMAKE_CURRENT_SOURCE_DIR}/identity/include
    ${CMAKE_CURRENT_SOURCE_DIR}/neuromorphic/include
    ${CMAKE_CURRENT_SOURCE_DIR}/swarm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cew/include
    ${CMAKE_CURRENT_SOURCE_DIR}/digital_twin/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(ARES_SOURCES
    main.cpp
    ares_edge_system.cpp
    common/src/data_structures.cpp
    common/src/logging.cpp
    common/src/config_manager.cpp
    core/quantum_resilient_core.cpp
    orchestrator/chronopath_engine.cpp
    optical_stealth/src/stealth_materials_controller.cpp
    countermeasures/src/last_man_standing_coordinator.cu
    countermeasures/src/chaos_induction_engine.cu
    countermeasures/src/self_destruct_protocol_simplified.cu
    cyber_em/src/em_cyber_controller.cu
    cyber_em/src/protocol_exploitation_engine.cu
    federated_learning/src/federated_learning_manager.cpp
    identity/src/did_manager.cpp
    neuromorphic/src/snn_processor.cpp
    swarm/src/swarm_coordinator.cpp
    cew/src/cew_controller.cpp
    digital_twin/src/digital_twin_manager.cpp
    backscatter/src/backscatter_communication_system.cpp
    backscatter/src/rf_energy_harvesting_system.cpp
)

# CUDA kernels
set(ARES_CUDA_KERNELS
    core/quantum_core.cu
    orchestrator/drpp_chronopath_engine.cu
    countermeasures/kernels/chaos_kernels.cu
    cyber_em/kernels/em_signal_processing_kernels.cu
    neuromorphic/kernels/neuromorphic_kernels.cu
    swarm/kernels/swarm_intelligence_kernels.cu
    cew/kernels/adaptive_jamming_kernel.cu
    cew/kernels/spectrum_waterfall_kernel.cu
    cew/kernels/threat_classifier_kernel.cu
    digital_twin/kernels/physics_simulation_kernels.cu
    digital_twin/kernels/state_sync_kernels.cu
    digital_twin/src/predictive_simulation_engine.cu
)

# Add the executable
add_executable(ares_edge_system ${ARES_SOURCES} ${ARES_CUDA_KERNELS})

# Link libraries
target_link_libraries(ares_edge_system PRIVATE
    Threads::Threads
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::curand
    CUDA::cusolver
    CUDA::cusparse
)

# Add cryptopp if available
find_library(CRYPTOPP_LIBRARY cryptopp)
if(CRYPTOPP_LIBRARY)
    target_link_libraries(ares_edge_system PRIVATE ${CRYPTOPP_LIBRARY})
endif()

if(OpenMP_FOUND)
    target_link_libraries(ares_edge_system PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set target properties
set_target_properties(ares_edge_system PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# Install command
install(TARGETS ares_edge_system DESTINATION bin)
