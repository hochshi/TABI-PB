#pragma once

#ifdef USE_CUDA_CC
#include <cuda_runtime.h>

#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "cuda_copy_sync_stats.h"
#include "cuda_state.h"

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
        const auto t0__ = std::chrono::steady_clock::now();                     \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        const auto t1__ = std::chrono::steady_clock::now();                     \
        tabipb_cuda_stats::record_device_sync(                                  \
            static_cast<std::uint64_t>(                                         \
                std::chrono::duration_cast<std::chrono::nanoseconds>(           \
                    t1__ - t0__).count()));                                     \
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
        const auto t0__ = std::chrono::steady_clock::now();                     \
        CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), (kind), (stream)));   \
        const auto t1__ = std::chrono::steady_clock::now();                     \
        tabipb_cuda_stats::record_memcpy(                                       \
            (kind), (bytes),                                                    \
            static_cast<std::uint64_t>(                                         \
                std::chrono::duration_cast<std::chrono::nanoseconds>(           \
                    t1__ - t0__).count()));                                     \
    } while (0)

#define CUDA_MEMCPY(dst, src, bytes, kind)                                      \
    do {                                                                        \
        const auto t0__ = std::chrono::steady_clock::now();                     \
        CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), (kind)));                  \
        const auto t1__ = std::chrono::steady_clock::now();                     \
        tabipb_cuda_stats::record_memcpy(                                       \
            (kind), (bytes),                                                    \
            static_cast<std::uint64_t>(                                         \
                std::chrono::duration_cast<std::chrono::nanoseconds>(           \
                    t1__ - t0__).count()));                                     \
    } while (0)

#define CUDA_STREAM_SYNC_AND_CHECK(stream)                                      \
    do {                                                                        \
        const auto t0__ = std::chrono::steady_clock::now();                     \
        CUDA_CHECK(cudaStreamSynchronize((stream)));                            \
        const auto t1__ = std::chrono::steady_clock::now();                     \
        tabipb_cuda_stats::record_stream_sync(                                  \
            static_cast<std::uint64_t>(                                         \
                std::chrono::duration_cast<std::chrono::nanoseconds>(           \
                    t1__ - t0__).count()));                                     \
    } while (0)

#define CUDA_EVENT_SYNC_AND_CHECK(event)                                        \
    do {                                                                        \
        const auto t0__ = std::chrono::steady_clock::now();                     \
        CUDA_CHECK(cudaEventSynchronize((event)));                              \
        const auto t1__ = std::chrono::steady_clock::now();                     \
        tabipb_cuda_stats::record_event_sync(                                   \
            static_cast<std::uint64_t>(                                         \
                std::chrono::duration_cast<std::chrono::nanoseconds>(           \
                    t1__ - t0__).count()));                                     \
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

inline bool cuda_pointer_is_device_accessible(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        // Query can fail for plain host pointers; clear sticky error and report not-device.
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
    return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

#else
#error "cuda_helpers.h requires USE_CUDA_CC"
#endif
