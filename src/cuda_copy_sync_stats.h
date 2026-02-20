#pragma once

#ifdef USE_CUDA_CC

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace tabipb_cuda_stats {

struct Snapshot {
    std::uint64_t h2d_bytes = 0;
    std::uint64_t d2h_bytes = 0;
    std::uint64_t d2d_bytes = 0;
    std::uint64_t other_bytes = 0;
    std::uint64_t h2d_calls = 0;
    std::uint64_t d2h_calls = 0;
    std::uint64_t d2d_calls = 0;
    std::uint64_t other_calls = 0;
    std::uint64_t device_sync_calls = 0;
    std::uint64_t stream_sync_calls = 0;
    std::uint64_t event_sync_calls = 0;
    std::uint64_t h2d_api_wall_ns = 0;
    std::uint64_t d2h_api_wall_ns = 0;
    std::uint64_t d2d_api_wall_ns = 0;
    std::uint64_t other_api_wall_ns = 0;
    std::uint64_t device_sync_wall_ns = 0;
    std::uint64_t stream_sync_wall_ns = 0;
    std::uint64_t event_sync_wall_ns = 0;
};

void record_memcpy(cudaMemcpyKind kind, std::size_t bytes,
                   std::uint64_t wall_ns = 0);
void record_device_sync(std::uint64_t wall_ns = 0);
void record_stream_sync(std::uint64_t wall_ns = 0);
void record_event_sync(std::uint64_t wall_ns = 0);
Snapshot snapshot();
void print_report_if_enabled();

}  // namespace tabipb_cuda_stats

#endif
