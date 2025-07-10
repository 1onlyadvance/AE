cmake_minimum_required(VERSION 3.18)

# Define the project name, version, and languages.
project(ARESEdgeSystem LANGUAGES CXX CUDA)

# Set the C++ and CUDA language standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find necessary packages.
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# Set specific flags for the CUDA compiler.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -DSKIP_CUDNN")

# Add include directories for the compiler to find header files.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator
    ${CMAKE_CURRENT_SOURCE_DIR}/optical_stealth/include
    ${CMAKE_CURRENT_SOURCE_DIR}/countermeasures/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cyber_em/include
    ${CMAKE_CURRENT_SOURCE_DIR}/federated_learning/include
    ${CUDA_INCLUDE_DIRS}
)

# List all the CUDA source files for the project.
set(CUDA_SOURCES
    core/quantum_core.cu
    swarm/kernels/auction_optimization_kernels.cu
    cew/kernels/adaptive_jamming_kernel.cu
    cew/kernels/spectrum_waterfall_kernel.cu
    cew/kernels/threat_classifier_kernel.cu
    digital_twin/kernels/physics_simulation_kernels.cu
    digital_twin/kernels/state_sync_kernels.cu
    neuromorphic/kernels/neuromorphic_simulation_kernels.cu
    neuromorphic/kernels/spike_encoding_kernel.cu
)

# Create the executable from the source files.
add_executable(ARESEdgeSystem ${CUDA_SOURCES})

# Link the required libraries to the executable.
target_link_libraries(ARESEdgeSystem PRIVATE Threads::Threads)
