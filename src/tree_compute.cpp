#include "tree_compute.h"
#include "tree_compute_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "tree_compute_backend_cuda.h"
#endif

void TreeCompute::run()
{
    const char* debug_env = std::getenv("TABIPB_DEBUG_PROGRESS");
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool debug_progress =
        require_all || (debug_env && std::strcmp(debug_env, "0") != 0);
    const char* verbose_env = std::getenv("TABIPB_DEBUG_TREE_RUN_VERBOSE");
    const bool debug_verbose =
        (verbose_env && std::strcmp(verbose_env, "0") != 0);

    TreeComputeBackendParams params;
    params.debug_progress = debug_progress;
    params.debug_verbose = debug_verbose;
    const auto run_view = view();

    if (debug_progress) {
        std::cerr << "[DEBUG] TreeCompute::run: upward_pass begin\n";
    }
    upward_pass();
    if (debug_progress) {
        std::cerr << "[DEBUG] TreeCompute::run: upward_pass end\n";
        std::cerr << "[DEBUG] TreeCompute::run: interaction loops begin\n";
    }

#ifdef USE_CUDA_CC
    tree_compute_run_cuda(*this, run_view, params);
#else
    tree_compute_run_cpu(*this, run_view, params);
#endif

    if (debug_progress) {
        std::cerr << "[DEBUG] TreeCompute::run: downward_pass begin\n";
    }
    downward_pass();
    if (debug_progress) {
        std::cerr << "[DEBUG] TreeCompute::run: downward_pass end\n";
    }
}
