# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.20)

# Define the project name and enable C++ and CUDA languages
project(ARES_Edge_System LANGUAGES CXX CUDA)

# Automatically find all .cpp and .cu files within the ares_edge_system directory
file(GLOB_RECURSE ARES_SOURCES
  "ares_edge_system/*.cpp"
  "ares_edge_system/*.cu"
)

# Add the executable target using the found source files
add_executable(
  ares_edge_system
  ${ARES_SOURCES}
)

# Link against the necessary CUDA libraries
target_link_libraries(ares_edge_system PRIVATE ${CUDA_LIBRARIES})