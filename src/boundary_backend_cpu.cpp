#include <cstdlib>
#include <cstring>

#include "boundary_element.h"

void BoundaryElement::matrix_vector_cpu_(double alpha, const double* __restrict potential_old,
                                         double beta,       double* __restrict potential_new)
{
    timers_.matrix_vector.start();

    const auto host = host_view();
    const double potential_coeff_1 = 0.5 * (1. +      params_.phys_eps_);
    const double potential_coeff_2 = 0.5 * (1. + 1. / params_.phys_eps_);

    std::memcpy(host.potential_temp, potential_new, host.potential_num * sizeof(double));
    std::memset(potential_new, 0, host.potential_num * sizeof(double));

    BoundaryElement::clear_cluster_charges();
    BoundaryElement::clear_cluster_potentials();

    elements_.compute_charges(potential_old);
    BoundaryElement::upward_pass();

    bool use_fused_pppc = false;
#ifdef USE_CUDA_CC
    {
        const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
        const bool require_all =
            !(require_all_env && std::strcmp(require_all_env, "0") == 0);
        const char* fused_env = std::getenv("TABIPB_CUDA_PPPC_FUSED");
        const char* require_fused_env = std::getenv("TABIPB_CUDA_REQUIRE_PPPC");
        const bool require_fused =
            require_all || (require_fused_env && std::strcmp(require_fused_env, "0") != 0);
        use_fused_pppc = require_fused || (fused_env && std::strcmp(fused_env, "0") != 0);
    }
#endif
    if (use_fused_pppc) {
        BoundaryElement::particle_cluster_interact_all(potential_new, potential_old, true);
    } else {
        BoundaryElement::particle_particle_interact_all(potential_new, potential_old);
        BoundaryElement::particle_cluster_interact_all(potential_new, potential_old, false);
    }
    BoundaryElement::cluster_cluster_interact_all(potential_new);

    BoundaryElement::downward_pass(potential_new);

    for (std::size_t i = 0; i < host.potential_num / 2; ++i)
        potential_new[i] = beta * host.potential_temp[i]
                + alpha * (potential_coeff_1 * potential_old[i] - potential_new[i]);

    for (std::size_t i = host.potential_num / 2; i < host.potential_num; ++i)
        potential_new[i] =  beta * host.potential_temp[i]
                + alpha * (potential_coeff_2 * potential_old[i] - potential_new[i]);

    timers_.matrix_vector.stop();
}
