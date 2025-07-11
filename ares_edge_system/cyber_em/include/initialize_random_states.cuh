/**
 * PROPRIETARY AND CONFIDENTIAL
 * 
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * Patent Pending - Application #63/826,067
 * 
 * This file defines the header for initialize_random_states for the ARES Edge System
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ares::cyber_em {

// Host function to call the kernel
void initialize_random_states(curandState* states = nullptr, uint32_t num_states = 1024, cudaStream_t stream = 0);

} // namespace ares::cyber_em
