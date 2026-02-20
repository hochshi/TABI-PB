#include "cuda_copy_sync_stats.h"

#ifdef USE_CUDA_CC

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace tabipb_cuda_stats {

namespace {

std::atomic<std::uint64_t> g_h2d_bytes{0};
std::atomic<std::uint64_t> g_d2h_bytes{0};
std::atomic<std::uint64_t> g_d2d_bytes{0};
std::atomic<std::uint64_t> g_other_bytes{0};
std::atomic<std::uint64_t> g_h2d_calls{0};
std::atomic<std::uint64_t> g_d2h_calls{0};
std::atomic<std::uint64_t> g_d2d_calls{0};
std::atomic<std::uint64_t> g_other_calls{0};
std::atomic<std::uint64_t> g_device_sync_calls{0};
std::atomic<std::uint64_t> g_stream_sync_calls{0};
std::atomic<std::uint64_t> g_event_sync_calls{0};
std::atomic<std::uint64_t> g_h2d_api_wall_ns{0};
std::atomic<std::uint64_t> g_d2h_api_wall_ns{0};
std::atomic<std::uint64_t> g_d2d_api_wall_ns{0};
std::atomic<std::uint64_t> g_other_api_wall_ns{0};
std::atomic<std::uint64_t> g_device_sync_wall_ns{0};
std::atomic<std::uint64_t> g_stream_sync_wall_ns{0};
std::atomic<std::uint64_t> g_event_sync_wall_ns{0};

std::once_flag g_register_once;

bool stats_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("TABIPB_CUDA_COPY_STATS");
        return (env && std::strcmp(env, "0") != 0);
    }();
    return enabled;
}

double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double ms(std::uint64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

void register_atexit_if_needed() {
    if (!stats_enabled()) {
        return;
    }
    std::call_once(g_register_once, []() {
        std::atexit([]() { print_report_if_enabled(); });
    });
}

}  // namespace

void record_memcpy(cudaMemcpyKind kind, std::size_t bytes,
                   std::uint64_t wall_ns) {
    if (!stats_enabled()) {
        return;
    }
    register_atexit_if_needed();
    const std::uint64_t b = static_cast<std::uint64_t>(bytes);
    switch (kind) {
        case cudaMemcpyHostToDevice:
            g_h2d_bytes.fetch_add(b, std::memory_order_relaxed);
            g_h2d_calls.fetch_add(1, std::memory_order_relaxed);
            g_h2d_api_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            break;
        case cudaMemcpyDeviceToHost:
            g_d2h_bytes.fetch_add(b, std::memory_order_relaxed);
            g_d2h_calls.fetch_add(1, std::memory_order_relaxed);
            g_d2h_api_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            break;
        case cudaMemcpyDeviceToDevice:
            g_d2d_bytes.fetch_add(b, std::memory_order_relaxed);
            g_d2d_calls.fetch_add(1, std::memory_order_relaxed);
            g_d2d_api_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            break;
        default:
            g_other_bytes.fetch_add(b, std::memory_order_relaxed);
            g_other_calls.fetch_add(1, std::memory_order_relaxed);
            g_other_api_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
            break;
    }
}

void record_device_sync(std::uint64_t wall_ns) {
    if (!stats_enabled()) {
        return;
    }
    register_atexit_if_needed();
    g_device_sync_calls.fetch_add(1, std::memory_order_relaxed);
    g_device_sync_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
}

void record_stream_sync(std::uint64_t wall_ns) {
    if (!stats_enabled()) {
        return;
    }
    register_atexit_if_needed();
    g_stream_sync_calls.fetch_add(1, std::memory_order_relaxed);
    g_stream_sync_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
}

void record_event_sync(std::uint64_t wall_ns) {
    if (!stats_enabled()) {
        return;
    }
    register_atexit_if_needed();
    g_event_sync_calls.fetch_add(1, std::memory_order_relaxed);
    g_event_sync_wall_ns.fetch_add(wall_ns, std::memory_order_relaxed);
}

Snapshot snapshot() {
    Snapshot s;
    s.h2d_bytes = g_h2d_bytes.load(std::memory_order_relaxed);
    s.d2h_bytes = g_d2h_bytes.load(std::memory_order_relaxed);
    s.d2d_bytes = g_d2d_bytes.load(std::memory_order_relaxed);
    s.other_bytes = g_other_bytes.load(std::memory_order_relaxed);
    s.h2d_calls = g_h2d_calls.load(std::memory_order_relaxed);
    s.d2h_calls = g_d2h_calls.load(std::memory_order_relaxed);
    s.d2d_calls = g_d2d_calls.load(std::memory_order_relaxed);
    s.other_calls = g_other_calls.load(std::memory_order_relaxed);
    s.device_sync_calls = g_device_sync_calls.load(std::memory_order_relaxed);
    s.stream_sync_calls = g_stream_sync_calls.load(std::memory_order_relaxed);
    s.event_sync_calls = g_event_sync_calls.load(std::memory_order_relaxed);
    s.h2d_api_wall_ns = g_h2d_api_wall_ns.load(std::memory_order_relaxed);
    s.d2h_api_wall_ns = g_d2h_api_wall_ns.load(std::memory_order_relaxed);
    s.d2d_api_wall_ns = g_d2d_api_wall_ns.load(std::memory_order_relaxed);
    s.other_api_wall_ns = g_other_api_wall_ns.load(std::memory_order_relaxed);
    s.device_sync_wall_ns = g_device_sync_wall_ns.load(std::memory_order_relaxed);
    s.stream_sync_wall_ns = g_stream_sync_wall_ns.load(std::memory_order_relaxed);
    s.event_sync_wall_ns = g_event_sync_wall_ns.load(std::memory_order_relaxed);
    return s;
}

void print_report_if_enabled() {
    if (!stats_enabled()) {
        return;
    }
    const Snapshot s = snapshot();
    const std::uint64_t sync_total =
        s.device_sync_calls + s.stream_sync_calls + s.event_sync_calls;

    std::cerr << "[CUDA_COPY_SYNC] memcpy calls:"
              << " H2D=" << s.h2d_calls
              << " D2H=" << s.d2h_calls
              << " D2D=" << s.d2d_calls
              << " other=" << s.other_calls << "\n";
    std::cerr << "[CUDA_COPY_SYNC] memcpy bytes:"
              << " H2D=" << s.h2d_bytes << " (" << mib(s.h2d_bytes) << " MiB)"
              << " D2H=" << s.d2h_bytes << " (" << mib(s.d2h_bytes) << " MiB)"
              << " D2D=" << s.d2d_bytes << " (" << mib(s.d2d_bytes) << " MiB)"
              << " other=" << s.other_bytes << " (" << mib(s.other_bytes) << " MiB)" << "\n";
    std::cerr << "[CUDA_COPY_SYNC] memcpy API wall time (ms):"
              << " H2D=" << ms(s.h2d_api_wall_ns)
              << " D2H=" << ms(s.d2h_api_wall_ns)
              << " D2D=" << ms(s.d2d_api_wall_ns)
              << " other=" << ms(s.other_api_wall_ns)
              << " total=" << ms(s.h2d_api_wall_ns + s.d2h_api_wall_ns +
                                 s.d2d_api_wall_ns + s.other_api_wall_ns) << "\n";
    std::cerr << "[CUDA_COPY_SYNC] sync calls:"
              << " device=" << s.device_sync_calls
              << " stream=" << s.stream_sync_calls
              << " event=" << s.event_sync_calls
              << " total=" << sync_total << "\n";
    std::cerr << "[CUDA_COPY_SYNC] sync API wall time (ms):"
              << " device=" << ms(s.device_sync_wall_ns)
              << " stream=" << ms(s.stream_sync_wall_ns)
              << " event=" << ms(s.event_sync_wall_ns)
              << " total=" << ms(s.device_sync_wall_ns + s.stream_sync_wall_ns +
                                 s.event_sync_wall_ns) << "\n";
    std::cerr << "[CUDA_COPY_SYNC] note: memcpy API time for async copies is enqueue time; "
              << "transfer wait shows up in sync API time.\n";
}

}  // namespace tabipb_cuda_stats

#endif
