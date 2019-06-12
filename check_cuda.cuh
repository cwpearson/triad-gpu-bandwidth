#pragma once

#include <cstdio>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    printf("%s@%d: CUDA Runtime Error: %s\n", file, line,
        cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);