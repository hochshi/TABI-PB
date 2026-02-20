#include "tree_compute.h"
#include "tree_compute_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "tree_compute_backend_cuda.h"
#endif

void TreeCompute::run()
{
    TreeComputeBackendParams params;
    const auto run_view = view();

    upward_pass();

#ifdef USE_CUDA_CC
    tree_compute_run_cuda(*this, run_view, params);
#else
    tree_compute_run_cpu(*this, run_view, params);
#endif

    downward_pass();
}
