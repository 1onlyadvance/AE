#pragma once

  // Include CUDA helpers
  #include "cuda_helpers.h"

  // Missing CUDA library types (when libraries not
  available)
  #ifdef SKIP_CUDNN
  typedef void* cudnnHandle_t;
  #endif

  #ifndef CUBLAS_V2_H_
  typedef void* cublasHandle_t;
  #endif

  #ifndef CUSOLVERDN_H_
  typedef void* cusolverDnHandle_t;
  #endif

  // Thread support for CUDA files
  #ifdef __CUDACC__
  namespace std {
      struct thread {
          thread() {}
          template<typename F, typename... Args>
          thread(F&& f, Args&&... args) {}
          void join() {}
      };
  }
  #endif

  // Additional math helpers
  inline __device__ float3 normalize(float3 v) {
      float len = length(v);
      return len > 0 ? v / len : make_float3(0, 0, 0);
  }

  Save it, then update CMakeLists.txt:

  nano CMakeLists.txt

  Update the CUDA flags section to include this new header
  globally:
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}
  -I${CMAKE_CURRENT_SOURCE_DIR}/common/include")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}
  --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include
  ares_cuda_fixes.h")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSKIP_CUDNN")
