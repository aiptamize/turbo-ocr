#pragma once
#include "turbo_ocr/common/errors.h"
#include <cuda_runtime.h>
#include <format>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      auto msg = std::format("CUDA Error at {}:{} - {}", __FILE__, __LINE__,  \
                             cudaGetErrorString(err));                         \
      std::cerr << msg << '\n';                                               \
      throw turbo_ocr::CudaError(msg);                                        \
    }                                                                          \
  } while (0)
