#include "solvation_backend_cuda.h"

#ifdef USE_CUDA_CC
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_helpers.h"
#include "solvation_energy_cuda.h"

bool SolvationEnergyCompute::validate_device_buffers_common_() const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!elements_.cuda_device_ready() || !molecule_.cuda_device_ready() ||
        !elem_interp_pts_.cuda_device_ready() || !mol_interp_pts_.cuda_device_ready()) {
        return false;
    }

    const auto& buf = device_buffers_;
    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = elem_interp_potential_.size();
    const std::size_t p_dx_num = elem_interp_potential_dx_.size();
    const std::size_t p_dy_num = elem_interp_potential_dy_.size();
    const std::size_t p_dz_num = elem_interp_potential_dz_.size();
    const std::size_t solv_eng_num = solv_eng_vec_.size();
    const std::size_t weights_up_num = mol_weights_.size();
    const std::size_t weights_down_num = elem_weights_.size();
    const std::size_t scratch_num = exact_idx_x_.size();

    if (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
        buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
        buf.solv_eng_num != solv_eng_num ||
        buf.weights_up_num != weights_up_num ||
        buf.weights_down_num != weights_down_num ||
        buf.scratch_num != scratch_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (p_dx_num > 0 && !buf.p_dx_dev) ||
        (p_dy_num > 0 && !buf.p_dy_dev) ||
        (p_dz_num > 0 && !buf.p_dz_dev) ||
        (solv_eng_num > 0 && !buf.solv_eng_dev) ||
        (weights_up_num > 0 && !buf.weights_up_dev) ||
        (weights_down_num > 0 && !buf.weights_down_dev) ||
        (scratch_num > 0 &&
         (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
          !buf.denominator_dev))) {
        return false;
    }
    return true;
}

bool SolvationEnergyCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_downward_pass_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool solvation_try_particle_particle_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         self_view.solv_eng && potential;
    if (!view_ok) {
        return false;
    }

    solvation_pp_cuda(elem_view.x, elem_view.y, elem_view.z,
                      elem_view.nx, elem_view.ny, elem_view.nz,
                      elem_view.area,
                      mol_view.particles_x, mol_view.particles_y,
                      mol_view.particles_z, mol_view.charge,
                      potential, params.potential_offset,
                      target_node_idxs[0], target_node_idxs[1],
                      source_node_idxs[0], source_node_idxs[1],
                      params.eps, params.kappa,
                      self_view.solv_eng, stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_particle_cluster_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& mol_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.solv_eng && potential;
    if (!view_ok) {
        return false;
    }

    solvation_pc_cuda(elem_view.x, elem_view.y, elem_view.z,
                      elem_view.nx, elem_view.ny, elem_view.nz,
                      elem_view.area,
                      mol_interp_view.interp_x, mol_interp_view.interp_y,
                      mol_interp_view.interp_z,
                      self_view.q,
                      potential, params.potential_offset,
                      source_node_idx,
                      params.num_mol_interp_pts_per_node,
                      params.num_mol_interp_charges_per_node,
                      target_node_idxs[0], target_node_idxs[1],
                      params.eps, params.kappa,
                      self_view.solv_eng, stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz;
    if (!view_ok) {
        return false;
    }

    solvation_cp_cuda(mol_view.particles_x, mol_view.particles_y,
                      mol_view.particles_z, mol_view.charge,
                      elem_interp_view.interp_x, elem_interp_view.interp_y,
                      elem_interp_view.interp_z,
                      self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
                      target_node_idx,
                      params.num_elem_interp_pts_per_node,
                      params.num_elem_interp_potentials_per_node,
                      source_node_idxs[0], source_node_idxs[1],
                      params.eps, params.kappa,
                      stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_cluster_cluster_cuda(
    const InterpolationPoints::View& mol_interp_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.q && self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz;
    if (!view_ok) {
        return false;
    }

    solvation_cc_cuda(mol_interp_view.interp_x, mol_interp_view.interp_y,
                      mol_interp_view.interp_z,
                      self_view.q,
                      elem_interp_view.interp_x, elem_interp_view.interp_y,
                      elem_interp_view.interp_z,
                      self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
                      target_node_idx, source_node_idx,
                      params.num_elem_interp_pts_per_node,
                      params.num_elem_interp_potentials_per_node,
                      params.num_mol_interp_pts_per_node,
                      params.num_mol_interp_charges_per_node,
                      params.eps, params.kappa,
                      stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.weights_up &&
                         self_view.exact_idx_x && self_view.exact_idx_y &&
                         self_view.exact_idx_z && self_view.denominator;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < source_tree.num_nodes(); ++node_idx) {
        const auto particle_idxs = source_tree.node_particle_idxs(node_idx);
        const std::size_t particle_start = particle_idxs[0];
        const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        if (num_particles > self_view.scratch_num) {
            return false;
        }
        solvation_up_cuda(
            mol_view.particles_x, mol_view.particles_y, mol_view.particles_z,
            mol_view.charge,
            mol_interp_view.interp_x, mol_interp_view.interp_y,
            mol_interp_view.interp_z,
            self_view.q,
            self_view.weights_up,
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

bool solvation_try_downward_pass_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const Tree& target_tree,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz &&
                         self_view.solv_eng && self_view.weights_down &&
                         potential;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < target_tree.num_nodes(); ++node_idx) {
        const auto particle_idxs = target_tree.node_particle_idxs(node_idx);
        const std::size_t particle_start = particle_idxs[0];
        const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        solvation_down_cuda(
            elem_view.x, elem_view.y, elem_view.z,
            elem_view.nx, elem_view.ny, elem_view.nz,
            elem_view.area,
            elem_interp_view.interp_x, elem_interp_view.interp_y,
            elem_interp_view.interp_z,
            self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
            potential, params.potential_offset,
            self_view.weights_down,
            node_idx,
            params.num_elem_interp_pts_per_node,
            params.num_elem_interp_potentials_per_node,
            particle_start,
            num_particles,
            self_view.solv_eng,
            stream);
        CUDA_CHECK_LAST_KERNEL();
    }
    CUDA_SYNC_AND_CHECK();
    return true;
}

void SolvationEnergyCompute::copyin_clusters_to_device_cuda_() const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);

    const double* q_ptr = mol_interp_charge_.data();
    const std::size_t q_num = mol_interp_charge_.size();

    const double* p_ptr = elem_interp_potential_.data();
    const double* p_dx_ptr = elem_interp_potential_dx_.data();
    const double* p_dy_ptr = elem_interp_potential_dy_.data();
    const double* p_dz_ptr = elem_interp_potential_dz_.data();

    const std::size_t p_num = elem_interp_potential_.size();
    const std::size_t p_dx_num = elem_interp_potential_dx_.size();
    const std::size_t p_dy_num = elem_interp_potential_dy_.size();
    const std::size_t p_dz_num = elem_interp_potential_dz_.size();

    const double* solv_eng_ptr = solv_eng_vec_.data();
    const std::size_t solv_eng_num = solv_eng_vec_.size();

    const double* weights_up_ptr = mol_weights_.data();
    const std::size_t weights_up_num = mol_weights_.size();
    const double* weights_down_ptr = elem_weights_.data();
    const std::size_t weights_down_num = elem_weights_.size();
    const std::size_t scratch_num = max_mol_particles_per_node_;

    auto& buf = device_buffers_;
    if (buf.ready &&
        (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
         buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
         buf.solv_eng_num != solv_eng_num ||
         buf.weights_up_num != weights_up_num ||
         buf.weights_down_num != weights_down_num ||
         buf.scratch_num != scratch_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        buf = DeviceBuffers{};
    }

    if (!buf.ready &&
        (q_num > 0 || p_num > 0 || solv_eng_num > 0 ||
         weights_up_num > 0 || weights_down_num > 0 || scratch_num > 0)) {
        if (q_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.q_dev, q_num * sizeof(double));
            buf.q_num = q_num;
        }
        if (p_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dev, p_num * sizeof(double));
            buf.p_num = p_num;
        }
        if (p_dx_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dx_dev, p_dx_num * sizeof(double));
            buf.p_dx_num = p_dx_num;
        }
        if (p_dy_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dy_dev, p_dy_num * sizeof(double));
            buf.p_dy_num = p_dy_num;
        }
        if (p_dz_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dz_dev, p_dz_num * sizeof(double));
            buf.p_dz_num = p_dz_num;
        }
        if (solv_eng_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.solv_eng_dev, solv_eng_num * sizeof(double));
            buf.solv_eng_num = solv_eng_num;
        }
        if (weights_up_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_up_dev, weights_up_num * sizeof(double));
            buf.weights_up_num = weights_up_num;
        }
        if (weights_down_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_down_dev, weights_down_num * sizeof(double));
            buf.weights_down_num = weights_down_num;
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
    if (q_num > 0 && buf.q_dev) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0 && buf.p_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dx_num > 0 && buf.p_dx_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dx_dev, p_dx_ptr, p_dx_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dy_num > 0 && buf.p_dy_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dy_dev, p_dy_ptr, p_dy_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dz_num > 0 && buf.p_dz_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dz_dev, p_dz_ptr, p_dz_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (solv_eng_num > 0 && buf.solv_eng_dev) {
        CUDA_MEMCPY_ASYNC(buf.solv_eng_dev, solv_eng_ptr,
                          solv_eng_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_up_num > 0 && buf.weights_up_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_up_dev, weights_up_ptr,
                          weights_up_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_down_num > 0 && buf.weights_down_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_down_dev, weights_down_ptr,
                          weights_down_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        if (!buf.ready ||
            (q_num > 0 && !buf.q_dev) ||
            (p_num > 0 && !buf.p_dev) ||
            (p_dx_num > 0 && !buf.p_dx_dev) ||
            (p_dy_num > 0 && !buf.p_dy_dev) ||
            (p_dz_num > 0 && !buf.p_dz_dev) ||
            (solv_eng_num > 0 && !buf.solv_eng_dev) ||
            (weights_up_num > 0 && !buf.weights_up_dev) ||
            (weights_down_num > 0 && !buf.weights_down_dev) ||
            (scratch_num > 0 &&
             (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
              !buf.denominator_dev))) {
            std::cerr << "[CUDA_SOLVATION] require_all set but device buffers not ready.\n";
            std::exit(1);
        }
    }
}

void SolvationEnergyCompute::delete_clusters_from_device_cuda_() const
{
    auto& buf = device_buffers_;
    const std::size_t solv_eng_num = solv_eng_vec_.size();

    if (buf.ready) {
        cudaStream_t stream = nullptr;
        if (solv_eng_num > 0 && buf.solv_eng_dev) {
            CUDA_MEMCPY_ASYNC(solv_eng_vec_.data(), buf.solv_eng_dev,
                              solv_eng_num * sizeof(double),
                              cudaMemcpyDeviceToHost, stream);
            CUDA_STREAM_SYNC_AND_CHECK(stream);
        }
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        buf = DeviceBuffers{};
        device_state_ = CudaDeviceState::HostOnly;
    }
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
