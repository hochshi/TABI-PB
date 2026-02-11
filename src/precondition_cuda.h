#pragma once

#include <cstddef>

extern "C" void precondition_diag_cuda(double* z, const double* r,
                                       std::size_t num,
                                       double coeff1, double coeff2,
                                       void* stream);
