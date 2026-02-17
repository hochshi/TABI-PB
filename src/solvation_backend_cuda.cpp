#include "solvation_energy_compute.h"

#ifdef USE_CUDA_CC
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

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

    if (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
        buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
        buf.solv_eng_num != solv_eng_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (p_dx_num > 0 && !buf.p_dx_dev) ||
        (p_dy_num > 0 && !buf.p_dy_dev) ||
        (p_dz_num > 0 && !buf.p_dz_dev) ||
        (solv_eng_num > 0 && !buf.solv_eng_dev)) {
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
    return validate_device_buffers_common_() &&
           (device_buffers_.weights_up_num ==
            static_cast<std::size_t>(num_mol_interp_pts_per_node_)) &&
           (device_buffers_.weights_up_num == 0 || device_buffers_.weights_up_dev != nullptr);
}

bool SolvationEnergyCompute::validate_device_buffers_downward_pass_() const
{
    return validate_device_buffers_common_() &&
           (device_buffers_.weights_down_num ==
            static_cast<std::size_t>(num_elem_interp_pts_per_node_)) &&
           (device_buffers_.weights_down_num == 0 || device_buffers_.weights_down_dev != nullptr) &&
           potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::try_particle_particle_interact_cuda_(std::size_t target_node_begin,
                                                                  std::size_t target_node_end,
                                                                  std::size_t source_node_begin,
                                                                  std::size_t source_node_end) const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto elem_dev = elements_.device_view();
    const auto mol_dev = molecule_.device_view();
    const auto self_dev = device_view();
    const double* potential_dev = potential_device_ptr_;
    const bool present_ok = validate_device_buffers_particle_particle_() &&
                            elem_dev.x && elem_dev.y && elem_dev.z &&
                            elem_dev.nx && elem_dev.ny && elem_dev.nz &&
                            elem_dev.area &&
                            mol_dev.particles_x && mol_dev.particles_y &&
                            mol_dev.particles_z && mol_dev.charge &&
                            self_dev.solv_eng && potential_dev;
    if (present_ok) {
        void* stream = nullptr;
        solvation_pp_cuda(elem_dev.x, elem_dev.y, elem_dev.z,
                          elem_dev.nx, elem_dev.ny, elem_dev.nz,
                          elem_dev.area,
                          mol_dev.particles_x, mol_dev.particles_y,
                          mol_dev.particles_z, mol_dev.charge,
                          potential_dev, potential_offset_,
                          target_node_begin, target_node_end,
                          source_node_begin, source_node_end,
                          eps_, kappa_,
                          self_dev.solv_eng, stream);
        CUDA_CHECK_LAST_KERNEL();
        return true;
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    return false;
}

bool SolvationEnergyCompute::try_particle_cluster_interact_cuda_(std::size_t target_node_begin,
                                                                 std::size_t target_node_end,
                                                                 std::size_t source_node_idx) const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto elem_dev = elements_.device_view();
    const auto mol_interp_dev = mol_interp_pts_.device_view();
    const auto self_dev = device_view();
    const double* potential_dev = potential_device_ptr_;
    const bool present_ok = validate_device_buffers_particle_cluster_() &&
                            elem_dev.x && elem_dev.y && elem_dev.z &&
                            elem_dev.nx && elem_dev.ny && elem_dev.nz &&
                            elem_dev.area &&
                            mol_interp_dev.interp_x && mol_interp_dev.interp_y &&
                            mol_interp_dev.interp_z &&
                            self_dev.q && self_dev.solv_eng && potential_dev;
    if (present_ok) {
        void* stream = nullptr;
        solvation_pc_cuda(elem_dev.x, elem_dev.y, elem_dev.z,
                          elem_dev.nx, elem_dev.ny, elem_dev.nz,
                          elem_dev.area,
                          mol_interp_dev.interp_x, mol_interp_dev.interp_y,
                          mol_interp_dev.interp_z,
                          self_dev.q,
                          potential_dev, potential_offset_,
                          source_node_idx,
                          num_mol_interp_pts_per_node_,
                          num_mol_interp_charges_per_node_,
                          target_node_begin, target_node_end,
                          eps_, kappa_,
                          self_dev.solv_eng, stream);
        CUDA_CHECK_LAST_KERNEL();
        return true;
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    return false;
}

bool SolvationEnergyCompute::try_cluster_particle_interact_cuda_(std::size_t target_node_idx,
                                                                 std::size_t source_node_begin,
                                                                 std::size_t source_node_end) const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto mol_dev = molecule_.device_view();
    const auto elem_interp_dev = elem_interp_pts_.device_view();
    const auto self_dev = device_view();
    const bool present_ok = validate_device_buffers_cluster_particle_() &&
                            mol_dev.particles_x && mol_dev.particles_y &&
                            mol_dev.particles_z && mol_dev.charge &&
                            elem_interp_dev.interp_x && elem_interp_dev.interp_y &&
                            elem_interp_dev.interp_z &&
                            self_dev.p && self_dev.p_dx &&
                            self_dev.p_dy && self_dev.p_dz;
    if (present_ok) {
        void* stream = nullptr;
        solvation_cp_cuda(mol_dev.particles_x, mol_dev.particles_y,
                          mol_dev.particles_z, mol_dev.charge,
                          elem_interp_dev.interp_x, elem_interp_dev.interp_y,
                          elem_interp_dev.interp_z,
                          self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
                          target_node_idx,
                          num_elem_interp_pts_per_node_,
                          num_elem_interp_potentials_per_node_,
                          source_node_begin, source_node_end,
                          eps_, kappa_, stream);
        CUDA_CHECK_LAST_KERNEL();
        return true;
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    return false;
}

bool SolvationEnergyCompute::try_cluster_cluster_interact_cuda_(std::size_t target_node_idx,
                                                                std::size_t source_node_idx) const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto mol_interp_dev = mol_interp_pts_.device_view();
    const auto elem_interp_dev = elem_interp_pts_.device_view();
    const auto self_dev = device_view();
    const bool present_ok = validate_device_buffers_cluster_cluster_() &&
                            mol_interp_dev.interp_x && mol_interp_dev.interp_y &&
                            mol_interp_dev.interp_z &&
                            elem_interp_dev.interp_x && elem_interp_dev.interp_y &&
                            elem_interp_dev.interp_z &&
                            self_dev.q &&
                            self_dev.p && self_dev.p_dx &&
                            self_dev.p_dy && self_dev.p_dz;
    if (present_ok) {
        void* stream = nullptr;
        solvation_cc_cuda(mol_interp_dev.interp_x, mol_interp_dev.interp_y,
                          mol_interp_dev.interp_z,
                          self_dev.q,
                          elem_interp_dev.interp_x, elem_interp_dev.interp_y,
                          elem_interp_dev.interp_z,
                          self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
                          target_node_idx, source_node_idx,
                          num_elem_interp_pts_per_node_,
                          num_elem_interp_potentials_per_node_,
                          num_mol_interp_pts_per_node_,
                          num_mol_interp_charges_per_node_,
                          eps_, kappa_, stream);
        CUDA_CHECK_LAST_KERNEL();
        return true;
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    return false;
}

bool SolvationEnergyCompute::try_upward_pass_cuda_() const
{
    std::vector<double> weights(num_mol_interp_pts_per_node_);
    for (int i = 0; i < num_mol_interp_pts_per_node_; ++i) {
        weights[i] = ((i % 2 == 0) ? 1 : -1);
        if (i == 0 || i == num_mol_interp_pts_per_node_ - 1) {
            weights[i] = ((i % 2 == 0) ? 1 : -1) * 0.5;
        }
    }

    auto& buf = device_buffers_;
    const std::size_t weights_num = static_cast<std::size_t>(weights.size());
    const std::size_t weights_bytes = weights_num * sizeof(double);
    if (buf.weights_up_num != 0 && buf.weights_up_num != weights_num) {
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        buf.weights_up_num = 0;
    }
    if (buf.weights_up_num == 0 && weights_num > 0) {
        CUDA_MALLOC_OR_DIE(&buf.weights_up_dev, weights_bytes);
        buf.weights_up_num = weights_num;
    }
    cudaStream_t stream = nullptr;
    if (weights_num > 0 && buf.weights_up_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_up_dev, weights.data(), weights_bytes,
                          cudaMemcpyHostToDevice, stream);
    }

    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto mol_dev = molecule_.device_view();
    const auto mol_interp_dev = mol_interp_pts_.device_view();
    const auto self_dev = device_view();
    const bool present_ok = validate_device_buffers_upward_pass_() &&
                            mol_dev.particles_x && mol_dev.particles_y &&
                            mol_dev.particles_z && mol_dev.charge &&
                            mol_interp_dev.interp_x && mol_interp_dev.interp_y &&
                            mol_interp_dev.interp_z &&
                            self_dev.q && self_dev.weights_up;
    if (present_ok) {
        std::size_t max_particles = 0;
        for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
            auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
            std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
            if (num_particles > max_particles) {
                max_particles = num_particles;
            }
        }
        if (max_particles > 0) {
            void* launch_stream = nullptr;
            int* exact_idx_x_dev = nullptr;
            int* exact_idx_y_dev = nullptr;
            int* exact_idx_z_dev = nullptr;
            double* denominator_dev = nullptr;
            CUDA_MALLOC_OR_DIE(&exact_idx_x_dev, max_particles * sizeof(int));
            CUDA_MALLOC_OR_DIE(&exact_idx_y_dev, max_particles * sizeof(int));
            CUDA_MALLOC_OR_DIE(&exact_idx_z_dev, max_particles * sizeof(int));
            CUDA_MALLOC_OR_DIE(&denominator_dev, max_particles * sizeof(double));
            for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
                auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
                std::size_t particle_start = particle_idxs[0];
                std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
                if (num_particles == 0) {
                    continue;
                }
                solvation_up_cuda(
                    mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                    mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                    self_dev.q,
                    self_dev.weights_up,
                    exact_idx_x_dev, exact_idx_y_dev, exact_idx_z_dev,
                    denominator_dev,
                    node_idx,
                    num_mol_interp_pts_per_node_,
                    num_mol_interp_charges_per_node_,
                    particle_start,
                    num_particles,
                    launch_stream);
                CUDA_CHECK_LAST_KERNEL();
            }
            CUDA_SYNC_AND_CHECK();
            CUDA_FREE_AND_NULL(exact_idx_x_dev);
            CUDA_FREE_AND_NULL(exact_idx_y_dev);
            CUDA_FREE_AND_NULL(exact_idx_z_dev);
            CUDA_FREE_AND_NULL(denominator_dev);
            if (buf.weights_up_dev) {
                CUDA_FREE_AND_NULL(buf.weights_up_dev);
                buf.weights_up_num = 0;
            }
            return true;
        }
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    if (buf.weights_up_dev) {
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        buf.weights_up_num = 0;
    }
    return false;
}

bool SolvationEnergyCompute::try_downward_pass_cuda_() const
{
    std::vector<double> weights(num_elem_interp_pts_per_node_);
    for (int i = 0; i < num_elem_interp_pts_per_node_; ++i) {
        weights[i] = ((i % 2 == 0) ? 1 : -1);
        if (i == 0 || i == num_elem_interp_pts_per_node_ - 1) {
            weights[i] = ((i % 2 == 0) ? 1 : -1) * 0.5;
        }
    }

    auto& buf = device_buffers_;
    const std::size_t weights_num = static_cast<std::size_t>(weights.size());
    const std::size_t weights_bytes = weights_num * sizeof(double);
    if (buf.weights_down_num != 0 && buf.weights_down_num != weights_num) {
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
        buf.weights_down_num = 0;
    }
    if (buf.weights_down_num == 0 && weights_num > 0) {
        CUDA_MALLOC_OR_DIE(&buf.weights_down_dev, weights_bytes);
        buf.weights_down_num = weights_num;
    }
    cudaStream_t stream = nullptr;
    if (weights_num > 0 && buf.weights_down_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_down_dev, weights.data(), weights_bytes,
                          cudaMemcpyHostToDevice, stream);
    }

    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const auto elem_dev = elements_.device_view();
    const auto elem_interp_dev = elem_interp_pts_.device_view();
    const auto self_dev = device_view();
    const double* potential_dev = potential_device_ptr_;
    const bool present_ok = validate_device_buffers_downward_pass_() &&
                            elem_dev.x && elem_dev.y && elem_dev.z &&
                            elem_dev.nx && elem_dev.ny && elem_dev.nz &&
                            elem_dev.area &&
                            elem_interp_dev.interp_x && elem_interp_dev.interp_y &&
                            elem_interp_dev.interp_z &&
                            self_dev.p && self_dev.p_dx &&
                            self_dev.p_dy && self_dev.p_dz &&
                            self_dev.solv_eng && self_dev.weights_down &&
                            potential_dev;
    if (present_ok) {
        void* launch_stream = nullptr;
        for (std::size_t node_idx = 0; node_idx < target_tree_.num_nodes(); ++node_idx) {
            auto particle_idxs = target_tree_.node_particle_idxs(node_idx);
            std::size_t particle_start = particle_idxs[0];
            std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
            if (num_particles == 0) {
                continue;
            }
            solvation_down_cuda(
                elem_dev.x, elem_dev.y, elem_dev.z,
                elem_dev.nx, elem_dev.ny, elem_dev.nz,
                elem_dev.area,
                elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
                self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
                potential_dev, potential_offset_,
                self_dev.weights_down,
                node_idx,
                num_elem_interp_pts_per_node_,
                num_elem_interp_potentials_per_node_,
                particle_start,
                num_particles,
                self_dev.solv_eng,
                launch_stream);
            CUDA_CHECK_LAST_KERNEL();
        }
        CUDA_SYNC_AND_CHECK();
        if (buf.weights_down_dev) {
            CUDA_FREE_AND_NULL(buf.weights_down_dev);
            buf.weights_down_num = 0;
        }
        return true;
    }
    if (require_all) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device pointers not present. "
                  << "Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }
    if (buf.weights_down_dev) {
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
        buf.weights_down_num = 0;
    }
    return false;
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

    auto& buf = device_buffers_;
    if (buf.ready &&
        (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
         buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
         buf.solv_eng_num != solv_eng_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        buf = DeviceBuffers{};
    }

    if (!buf.ready && (q_num > 0 || p_num > 0 || solv_eng_num > 0)) {
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

    CUDA_SYNC_AND_CHECK();
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        if (!buf.ready ||
            (q_num > 0 && !buf.q_dev) ||
            (p_num > 0 && !buf.p_dev) ||
            (p_dx_num > 0 && !buf.p_dx_dev) ||
            (p_dy_num > 0 && !buf.p_dy_dev) ||
            (p_dz_num > 0 && !buf.p_dz_dev) ||
            (solv_eng_num > 0 && !buf.solv_eng_dev)) {
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
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
        buf = DeviceBuffers{};
        device_state_ = CudaDeviceState::HostOnly;
    }
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
