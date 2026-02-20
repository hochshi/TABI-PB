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
    const char* gmres_env = std::getenv("TABIPB_CUDA_GMRES");
    const char* require_env = std::getenv("TABIPB_CUDA_REQUIRE_GMRES");
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool require_gmres = require_all ||
                               (require_env && std::strcmp(require_env, "0") != 0);
    const bool use_cuda = require_gmres || (gmres_env && std::strcmp(gmres_env, "0") != 0);
    if (use_cuda) {
        return gmres_cuda_(view);
    }
#endif

    return gmres_cpu_(view);
}
