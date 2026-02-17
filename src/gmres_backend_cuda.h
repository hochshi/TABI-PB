#pragma once

#ifdef USE_CUDA_CC
void gmres_update_cuda(long int i, long int n, double* x,
                       double* v_dev, long int ldv,
                       double* x_dev, double* h_dev, long int ldh,
                       double* s_dev, void* stream);
#endif
