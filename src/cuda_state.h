#pragma once

#ifdef USE_CUDA_CC
enum class CudaDeviceState {
    HostOnly,
    DeviceMapped
};
#endif
