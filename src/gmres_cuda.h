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
extern "C" void gmres_cuda_basis(long i, long n,
                                 double* h_col, double* v, std::size_t ldv,
                                 double* w, void* stream);
extern "C" void gmres_cuda_dtrsv_upper(const double* a, std::size_t lda,
                                       double* x, std::size_t n, void* stream);
extern "C" void gmres_cuda_dgemv(const double* a, std::size_t lda,
                                 const double* x, double* y,
                                 std::size_t m, std::size_t n, void* stream);
extern "C" void gmres_cuda_apply_prev_givens(double* h, std::size_t ldh,
                                             long i, long restrt, void* stream);
extern "C" void gmres_cuda_apply_givens(double* h, std::size_t ldh,
                                        double* s, long i, long restrt,
                                        double* resid_out, void* stream);

#endif
