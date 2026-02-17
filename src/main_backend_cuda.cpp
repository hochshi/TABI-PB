#include "main_backend.h"

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"

void main_backend_initialize_runtime() {
  // Initialize CUDA runtime/context before any module allocations.
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
}
#endif
