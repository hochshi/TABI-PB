#pragma once

#ifdef USE_CUDA_CC
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "cuda_state.h"

#ifdef OPENACC_ENABLED
#include <openacc.h>
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA] %s:%d %s failed: %s\n",                \
                         __FILE__, __LINE__, #call, cudaGetErrorString(err__)); \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST_KERNEL()                                                \
    do {                                                                        \
        cudaError_t err__ = cudaGetLastError();                                 \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA] %s:%d kernel launch failed: %s\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CUDA_SYNC_AND_CHECK()                                                   \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

#define CUDA_MALLOC_OR_DIE(ptr, bytes)                                          \
    do {                                                                        \
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), (bytes)));         \
    } while (0)

#define CUDA_FREE_AND_NULL(ptr)                                                 \
    do {                                                                        \
        if ((ptr) != nullptr) {                                                 \
            CUDA_CHECK(cudaFree(ptr));                                          \
            (ptr) = nullptr;                                                    \
        }                                                                       \
    } while (0)

#define CUDA_MEMCPY_ASYNC(dst, src, bytes, kind, stream)                        \
    do {                                                                        \
        CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), (kind), (stream)));   \
    } while (0)

#define CUDA_ZERO_ASYNC(ptr, bytes, stream)                                     \
    do {                                                                        \
        CUDA_CHECK(cudaMemsetAsync((ptr), 0, (bytes), (stream)));               \
    } while (0)

#define CUDA_REQUIRE_PTR(ptr, name)                                             \
    do {                                                                        \
        if ((ptr) == nullptr) {                                                 \
            std::fprintf(stderr, "[CUDA] required ptr %s is null (%s:%d)\n",    \
                         (name), __FILE__, __LINE__);                           \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CUDA_REQUIRE_SIZE(bytes, expected, name)                                \
    do {                                                                        \
        if ((bytes) != (expected)) {                                            \
            std::fprintf(stderr,                                                \
                         "[CUDA] size mismatch for %s: got %zu expected %zu "   \
                         "(%s:%d)\n",                                           \
                         (name),                                                \
                         static_cast<std::size_t>(bytes),                       \
                         static_cast<std::size_t>(expected),                    \
                         __FILE__, __LINE__);                                   \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#ifdef OPENACC_ENABLED
inline bool CUDA_ACC_IS_PRESENT(const void* host_ptr, std::size_t bytes) {
    if (bytes == 0) {
        return false;
    }
    return acc_is_present(const_cast<void*>(host_ptr), bytes) != 0;
}

inline void CUDA_ACC_UNMAP_IF_PRESENT(const void* host_ptr, std::size_t bytes) {
    if (CUDA_ACC_IS_PRESENT(host_ptr, bytes)) {
        acc_unmap_data(const_cast<void*>(host_ptr));
    }
}

inline void CUDA_ACC_MAP_CONST(const void* host_ptr, void* dev_ptr,
                               std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    acc_map_data(const_cast<void*>(host_ptr), dev_ptr, bytes);
}
#endif

#else
#error "cuda_helpers.h requires USE_CUDA_CC"
#endif
