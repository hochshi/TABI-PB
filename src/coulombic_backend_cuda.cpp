#include "coulombic_backend_cuda.h"

#ifdef USE_CUDA_CC
#include <cstdlib>
#include <cstring>

#include "coulombic_energy_cuda.h"
#include "cuda_helpers.h"

bool CoulombicEnergyCompute::validate_device_buffers_common_() const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!molecule_.cuda_device_ready() || !mol_interp_pts_.cuda_device_ready()) {
        return false;
    }

    const auto& buf = device_buffers_;
    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = mol_interp_potential_.size();
    const std::size_t coul_eng_num = coul_eng_vec_.size();
    const std::size_t weights_num = mol_weights_.size();
    const std::size_t scratch_num = exact_idx_x_.size();

    if (buf.q_num != q_num || buf.p_num != p_num ||
        buf.coul_eng_num != coul_eng_num || buf.weights_num != weights_num ||
        buf.scratch_num != scratch_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (coul_eng_num > 0 && !buf.coul_eng_dev) ||
        (weights_num > 0 && !buf.weights_dev) ||
        (scratch_num > 0 &&
         (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
          !buf.denominator_dev))) {
        return false;
    }
    return true;
}

bool CoulombicEnergyCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}

bool coulombic_try_particle_particle_cuda(
    const Molecule::View& mol_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params,
    void* stream) {
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         self_view.coul_eng;
    if (!view_ok) {
        return false;
    }

    coulombic_pp_cuda(
        mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
        target_node_idxs[0],
        target_node_idxs[1],
        source_node_idxs[0],
        source_node_idxs[1],
        params.eps_solute,
        self_view.coul_eng,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool coulombic_try_particle_cluster_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params,
    void* stream) {
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.coul_eng;
    if (!view_ok) {
        return false;
    }

    coulombic_pc_cuda(
        mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
        mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
        self_view.q,
        source_node_idx,
        params.num_mol_interp_pts_per_node,
        params.num_mol_interp_charges_per_node,
        target_node_idxs[0],
        target_node_idxs[1],
        params.eps_solute,
        self_view.coul_eng,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool coulombic_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params,
    void* stream) {
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.p;
    if (!view_ok) {
        return false;
    }

    coulombic_cp_cuda(
        mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
        self_view.p,
        mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
        target_node_idx,
        params.num_mol_interp_pts_per_node,
        params.num_mol_interp_potentials_per_node,
        source_node_idxs[0],
        source_node_idxs[1],
        params.eps_solute,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool coulombic_try_cluster_cluster_cuda(
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params,
    void* stream) {
    const bool view_ok = mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.p;
    if (!view_ok) {
        return false;
    }

    coulombic_cc_cuda(
        mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
        self_view.q, self_view.p,
        target_node_idx,
        source_node_idx,
        params.num_mol_interp_pts_per_node,
        params.num_mol_interp_charges_per_node,
        params.num_mol_interp_potentials_per_node,
        params.eps_solute,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool coulombic_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const CoulombicBackendParams& params,
    void* stream) {
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.weights &&
                         self_view.exact_idx_x && self_view.exact_idx_y &&
                         self_view.exact_idx_z && self_view.denominator;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < source_tree.num_nodes(); ++node_idx) {
        auto particle_idxs = source_tree.node_particle_idxs(node_idx);
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        coulombic_up_cuda(
            mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
            mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
            self_view.q,
            self_view.weights,
            self_view.exact_idx_x, self_view.exact_idx_y, self_view.exact_idx_z,
            self_view.denominator,
            node_idx,
            params.num_mol_interp_pts_per_node,
            params.num_mol_interp_charges_per_node,
            particle_start,
            num_particles,
            stream);
        CUDA_CHECK_LAST_KERNEL();
    }
    CUDA_SYNC_AND_CHECK();
    return true;
}

void CoulombicEnergyCompute::copyin_clusters_to_device_cuda_() const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        !(require_all_env && std::strcmp(require_all_env, "0") == 0);

    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();

    const double* p_ptr = mol_interp_potential_.data();
    std::size_t p_num   = mol_interp_potential_.size();

    double* coul_eng_ptr = coul_eng_vec_.data();
    std::size_t coul_eng_num   = coul_eng_vec_.size();

    const double* weights_ptr = mol_weights_.data();
    std::size_t weights_num   = mol_weights_.size();
    std::size_t scratch_num   = max_mol_particles_per_node_;

    auto& buf = device_buffers_;
    if (buf.ready &&
        (buf.q_num != q_num || buf.p_num != p_num ||
         buf.coul_eng_num != coul_eng_num || buf.weights_num != weights_num ||
         buf.scratch_num != scratch_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.coul_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        buf = DeviceBuffers{};
    }

    if (!buf.ready &&
        (q_num > 0 || p_num > 0 || coul_eng_num > 0 || weights_num > 0 || scratch_num > 0)) {
        if (q_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.q_dev, q_num * sizeof(double));
            buf.q_num = q_num;
        }
        if (p_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dev, p_num * sizeof(double));
            buf.p_num = p_num;
        }
        if (coul_eng_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.coul_eng_dev, coul_eng_num * sizeof(double));
            buf.coul_eng_num = coul_eng_num;
        }
        if (weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_dev, weights_num * sizeof(double));
            buf.weights_num = weights_num;
        }
        if (scratch_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_x_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_y_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_z_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.denominator_dev, scratch_num * sizeof(double));
            buf.scratch_num = scratch_num;
        }
        buf.ready = true;
    }

    cudaStream_t stream = nullptr;

    if (q_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (coul_eng_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.coul_eng_dev, coul_eng_ptr, coul_eng_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.weights_dev, weights_ptr, weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        const bool ok =
            (q_num == 0 || (buf.q_dev && buf.q_num == q_num)) &&
            (p_num == 0 || (buf.p_dev && buf.p_num == p_num)) &&
            (coul_eng_num == 0 || (buf.coul_eng_dev && buf.coul_eng_num == coul_eng_num)) &&
            (weights_num == 0 || (buf.weights_dev && buf.weights_num == weights_num)) &&
            (scratch_num == 0 || (buf.exact_idx_x_dev && buf.exact_idx_y_dev &&
                                  buf.exact_idx_z_dev && buf.denominator_dev &&
                                  buf.scratch_num == scratch_num));
        if (!ok) {
            std::fprintf(stderr,
                         "[CUDA] CoulombicEnergyCompute missing or mismatched device buffers "
                         "under TABIPB_CUDA_REQUIRE_ALL=1\n");
            std::abort();
        }
    }
}

void CoulombicEnergyCompute::delete_clusters_from_device_cuda_() const
{
    auto& buf = device_buffers_;
    const std::size_t coul_eng_num = coul_eng_vec_.size();
    double* coul_eng_ptr = coul_eng_vec_.data();

    if (buf.coul_eng_dev && coul_eng_num > 0) {
        cudaStream_t stream = nullptr;
        CUDA_MEMCPY_ASYNC(coul_eng_ptr, buf.coul_eng_dev, coul_eng_num * sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
        CUDA_SYNC_AND_CHECK();
    }

    CUDA_FREE_AND_NULL(buf.q_dev);
    CUDA_FREE_AND_NULL(buf.p_dev);
    CUDA_FREE_AND_NULL(buf.coul_eng_dev);
    CUDA_FREE_AND_NULL(buf.weights_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
    CUDA_FREE_AND_NULL(buf.denominator_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
