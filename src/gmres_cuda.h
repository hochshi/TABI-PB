#ifndef H_TABIPB_GMRES_CUDA_H
#define H_TABIPB_GMRES_CUDA_H

#include <cstddef>

extern "C" void gmres_cuda_copy(double* dst, const double* src, std::size_t n, void* stream);
extern "C" void gmres_cuda_dscal(double* x, double alpha, std::size_t n, void* stream);
extern "C" void gmres_cuda_daxpy(double* y, const double* x, double alpha, std::size_t n, void* stream);
extern "C" double gmres_cuda_ddot(const double* x, const double* y, std::size_t n,
                                  double* partials, std::size_t partials_len, void* stream);
extern "C" double gmres_cuda_dnrm2(const double* x, std::size_t n,
                                   double* partials, std::size_t partials_len, void* stream);
extern "C" void gmres_cuda_dtrsv_upper(const double* a, std::size_t lda,
                                       double* x, std::size_t n, void* stream);
extern "C" void gmres_cuda_dgemv(const double* a, std::size_t lda,
                                 const double* x, double* y,
                                 std::size_t m, std::size_t n, void* stream);

#endif
