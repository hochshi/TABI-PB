#include <cstdlib>
#include <cstring>

#include "boundary_element.h"

int BoundaryElement::gmres_(long int n, const double* b, double* x, long int restrt,
                     double* work, long int ldw, double* h, long int ldh,
                     long int& iter, double& resid)
{
    GmresView view;
    view.n = n;
    view.b = b;
    view.x = x;
    view.restrt = restrt;
    view.work = work;
    view.ldw = ldw;
    view.h = h;
    view.ldh = ldh;
    view.iter = &iter;
    view.residual = &resid;

#ifdef USE_CUDA_CC
    return gmres_cuda_(view);
#endif

    return gmres_cpu_(view);
}
