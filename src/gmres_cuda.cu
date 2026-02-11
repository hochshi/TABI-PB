#include "gmres_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>

namespace {

__global__ void dscal_kernel(double* x, double alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= alpha;
}

__global__ void daxpy_kernel(double* y, const double* x, double alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += alpha * x[idx];
}

__global__ void axpy_neg_kernel(double* y, const double* x, const double* alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        double a = *alpha;
        y[idx] -= a * x[idx];
    }
}

__global__ void scale_copy_kernel(double* dst, const double* src, const double* alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        double a = *alpha;
        dst[idx] = src[idx] / a;
    }
}

__device__ inline void drot_device(double& dx, double& dy, double c, double s)
{
    double dtemp = c * dx + s * dy;
    dy = c * dy - s * dx;
    dx = dtemp;
}

__device__ inline void drotg_device(double da, double db, double& c, double& s)
{
    double roe = db;
    if (fabs(da) > fabs(db)) roe = da;
    double scale = fabs(da) + fabs(db);
    if (scale != 0.0) {
        double d1 = da / scale;
        double d2 = db / scale;
        double r = scale * sqrt(d1 * d1 + d2 * d2) * (roe >= 0.0 ? 1.0 : -1.0);
        c = da / r;
        s = db / r;
    } else {
        c = 1.0;
        s = 0.0;
    }
}

__global__ void apply_prev_givens_kernel(double* h, std::size_t ldh, int i, int restrt)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int k = 0; k < i; ++k) {
        double c = h[k + static_cast<std::size_t>(restrt) * ldh];
        double s = h[k + static_cast<std::size_t>(restrt + 1) * ldh];
        double dx = h[k + static_cast<std::size_t>(i) * ldh];
        double dy = h[k + 1 + static_cast<std::size_t>(i) * ldh];
        drot_device(dx, dy, c, s);
        h[k + static_cast<std::size_t>(i) * ldh] = dx;
        h[k + 1 + static_cast<std::size_t>(i) * ldh] = dy;
    }
}

__global__ void apply_givens_kernel(double* h, std::size_t ldh,
                                    double* s, int i, int restrt,
                                    double* resid_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double c = 1.0;
    double srot = 0.0;
    double hii = h[i + static_cast<std::size_t>(i) * ldh];
    double hip1 = h[i + 1 + static_cast<std::size_t>(i) * ldh];
    drotg_device(hii, hip1, c, srot);
    h[i + static_cast<std::size_t>(restrt) * ldh] = c;
    h[i + static_cast<std::size_t>(restrt + 1) * ldh] = srot;
    double dx = hii;
    double dy = hip1;
    drot_device(dx, dy, c, srot);
    h[i + static_cast<std::size_t>(i) * ldh] = dx;
    h[i + 1 + static_cast<std::size_t>(i) * ldh] = dy;
    double s_i = s[i];
    double s_ip1 = s[i + 1];
    drot_device(s_i, s_ip1, c, srot);
    s[i] = s_i;
    s[i + 1] = s_ip1;
    if (resid_out) resid_out[0] = fabs(s_ip1);
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

namespace {
inline cublasHandle_t get_cublas_handle(cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return nullptr;
        }
        initialized = true;
    }
    cublasSetStream(handle, stream);
    return handle;
}
} // namespace

extern "C" void gmres_cuda_copy(double* dst, const double* src, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice, cuda_stream);
}

extern "C" void gmres_cuda_dscal(double* x, double alpha, std::size_t n, void* stream)
{
    constexpr int kBlock = 256;
    const int grid = grid_for(n, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    dscal_kernel<<<grid, kBlock, 0, cuda_stream>>>(x, alpha, n);
}

extern "C" void gmres_cuda_daxpy(double* y, const double* x, double alpha, std::size_t n, void* stream)
{
    constexpr int kBlock = 256;
    const int grid = grid_for(n, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    daxpy_kernel<<<grid, kBlock, 0, cuda_stream>>>(y, x, alpha, n);
}

extern "C" double gmres_cuda_ddot(const double* x, const double* y, std::size_t n,
                                  double* partials, std::size_t partials_len, void* stream)
{
    (void)partials;
    (void)partials_len;
    if (n == 0) return 0.0;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return 0.0;
    if (partials != nullptr && partials_len >= 1) {
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasDdot(handle, static_cast<int>(n), x, 1, y, 1, partials);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        return 0.0;
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    double result = 0.0;
    cublasDdot(handle, static_cast<int>(n), x, 1, y, 1, &result);
    return result;
}

extern "C" double gmres_cuda_dnrm2(const double* x, std::size_t n,
                                   double* partials, std::size_t partials_len, void* stream)
{
    (void)partials;
    (void)partials_len;
    if (n == 0) return 0.0;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return 0.0;
    if (partials != nullptr && partials_len >= 1) {
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasDnrm2(handle, static_cast<int>(n), x, 1, partials);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        return 0.0;
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    double result = 0.0;
    cublasDnrm2(handle, static_cast<int>(n), x, 1, &result);
    return result;
}

extern "C" void gmres_cuda_dtrsv_upper(const double* a, std::size_t lda,
                                       double* x, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return;
    cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                static_cast<int>(n), a, static_cast<int>(lda), x, 1);
}

extern "C" void gmres_cuda_dgemv(const double* a, std::size_t lda,
                                 const double* x, double* y,
                                 std::size_t m, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return;
    const double alpha = 1.0;
    const double beta = 1.0;
    cublasDgemv(handle, CUBLAS_OP_N,
                static_cast<int>(m), static_cast<int>(n),
                &alpha, a, static_cast<int>(lda),
                x, 1, &beta, y, 1);
}

extern "C" void gmres_cuda_basis(long i, long n,
                                 double* h_col, double* v, std::size_t ldv,
                                 double* w, void* stream)
{
    if (i <= 0 || n <= 0) return;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return;
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    for (long k = 0; k < i; ++k) {
        double* h_entry = h_col + k;
        const double* v_k = v + static_cast<std::size_t>(k) * ldv;
        cublasDdot(handle, static_cast<int>(n), w, 1, v_k, 1, h_entry);
        constexpr int kBlock = 256;
        const int grid = grid_for(static_cast<std::size_t>(n), kBlock);
        axpy_neg_kernel<<<grid, kBlock, 0, cuda_stream>>>(w, v_k, h_entry, static_cast<std::size_t>(n));
    }
    double* h_diag = h_col + i;
    cublasDnrm2(handle, static_cast<int>(n), w, 1, h_diag);
    constexpr int kBlock = 256;
    const int grid = grid_for(static_cast<std::size_t>(n), kBlock);
    scale_copy_kernel<<<grid, kBlock, 0, cuda_stream>>>(v + static_cast<std::size_t>(i) * ldv,
                                                        w, h_diag, static_cast<std::size_t>(n));
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
}

extern "C" void gmres_cuda_apply_prev_givens(double* h, std::size_t ldh,
                                             long i, long restrt, void* stream)
{
    if (i <= 0) return;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    apply_prev_givens_kernel<<<1, 1, 0, cuda_stream>>>(h, ldh,
                                                       static_cast<int>(i),
                                                       static_cast<int>(restrt));
}

extern "C" void gmres_cuda_apply_givens(double* h, std::size_t ldh,
                                        double* s, long i, long restrt,
                                        double* resid_out, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    apply_givens_kernel<<<1, 1, 0, cuda_stream>>>(h, ldh, s,
                                                  static_cast<int>(i),
                                                  static_cast<int>(restrt),
                                                  resid_out);
}
