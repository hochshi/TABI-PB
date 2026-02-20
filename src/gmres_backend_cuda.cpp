#include "gmres_backend_cuda.h"
#include "boundary_element.h"

#ifdef USE_CUDA_CC
#include "gmres_cuda.h"

int BoundaryElement::gmres_cuda_(const GmresView& view)
{
    return gmres_impl_(view, true);
}

void gmres_update_cuda(long int i, long int n, double* x,
                       double* v_dev, long int ldv,
                       double* x_dev, double* h_dev, long int ldh,
                       double* s_dev, void* stream)
{
    if (i <= 0) return;
    (void)x;
    gmres_cuda_dtrsv_upper(h_dev, static_cast<std::size_t>(ldh),
                           s_dev, static_cast<std::size_t>(i), stream);
    gmres_cuda_dgemv(v_dev, static_cast<std::size_t>(ldv),
                     s_dev, x_dev,
                     static_cast<std::size_t>(n), static_cast<std::size_t>(i), stream);
}
#endif
