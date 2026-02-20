#include "output.h"

#ifdef USE_CUDA_CC
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_helpers.h"
#include "output_cuda.h"

bool Output::validate_device_buffers_compute_coulombic_energy_() const {
    const auto mol_dev = molecule_.device_view();
    return molecule_.cuda_device_ready() &&
           mol_dev.particles_x &&
           mol_dev.particles_y &&
           mol_dev.particles_z &&
           mol_dev.charge &&
           mol_dev.num_particles == molecule_.num();
}

bool Output::validate_device_buffers_compute_solvation_energy_() const {
    const auto& buf = device_buffers_;
    return elements_.cuda_device_ready() &&
           molecule_.cuda_device_ready() &&
           device_state_ == CudaDeviceState::DeviceMapped &&
           buf.ready &&
           buf.potential_dev &&
           buf.potential_num == potential_.size();
}

void Output::copyin_potential_to_device_cuda_(const double* potential_ptr,
                                              std::size_t potential_num) const {
    auto& buf = device_buffers_;
    if (buf.potential_num != 0 && buf.potential_num != potential_num) {
        CUDA_FREE_AND_NULL(buf.potential_dev);
        buf.potential_num = 0;
        buf.ready = false;
    }
    if (buf.potential_num == 0 && potential_num > 0) {
        CUDA_MALLOC_OR_DIE(&buf.potential_dev, potential_num * sizeof(double));
        buf.potential_num = potential_num;
    }

    cudaStream_t stream = nullptr;
    if (potential_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.potential_dev, potential_ptr,
                          potential_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    CUDA_SYNC_AND_CHECK();

    buf.ready = true;
    device_state_ = CudaDeviceState::DeviceMapped;

    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    if (require_all && potential_num > 0) {
        if (!buf.potential_dev || !buf.ready || buf.potential_num != potential_num) {
            std::cerr << "[CUDA_OUTPUT] require_all set but potential buffer not ready. "
                      << "Aborting to avoid OpenACC fallback.\n";
            std::exit(1);
        }
    }
}

void Output::cleanup_potential_device_buffer_cuda_() const {
    auto& buf = device_buffers_;
    CUDA_FREE_AND_NULL(buf.potential_dev);
    buf.potential_num = 0;
    buf.ready = false;
    device_state_ = CudaDeviceState::HostOnly;
}

bool Output::try_compute_coulombic_energy_cuda_(double& coulombic_energy) const {
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool present_ok = validate_device_buffers_compute_coulombic_energy_();
    if (!present_ok) {
        if (require_all) {
            std::cerr << "[CUDA_OUTPUT] require_all set but device pointers not present. "
                      << "Aborting to avoid OpenACC fallback.\n";
            std::exit(1);
        }
        return false;
    }

    const auto mol_dev = molecule_.device_view();
    const std::size_t num_atoms = molecule_.num();
    const double epsp = params_.phys_eps_solute_;

    cudaStream_t stream = nullptr;
    double* energy_dev = nullptr;
    auto check = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA_OUTPUT] " << what << " failed: "
                      << cudaGetErrorString(err) << "\n";
            std::exit(1);
        }
    };

    check(cudaMalloc(&energy_dev, sizeof(double)), "cudaMalloc coulombic_energy");
    check(cudaMemsetAsync(energy_dev, 0, sizeof(double), stream),
          "cudaMemsetAsync coulombic_energy");

    output_coulombic_cuda(mol_dev.particles_x, mol_dev.particles_y,
                          mol_dev.particles_z, mol_dev.charge,
                          num_atoms, epsp, energy_dev, stream);
    CUDA_CHECK_LAST_KERNEL();

    const auto copy_t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(&coulombic_energy, energy_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync coulombic_energy");
    const auto copy_t1 = std::chrono::steady_clock::now();
    tabipb_cuda_stats::record_memcpy(
        cudaMemcpyDeviceToHost, sizeof(double),
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(copy_t1 - copy_t0).count()));
    CUDA_STREAM_SYNC_AND_CHECK(stream);
    cudaFree(energy_dev);

    return true;
}

bool Output::try_compute_solvation_energy_cuda_(double& solvation_energy) const {
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool present_ok = validate_device_buffers_compute_solvation_energy_();
    if (!present_ok) {
        if (require_all) {
            std::cerr << "[CUDA_OUTPUT] require_all set but device pointers not present. "
                      << "Aborting to avoid OpenACC fallback.\n";
            std::exit(1);
        }
        return false;
    }

    const auto elem_dev = elements_.device_view();
    const auto mol_dev = molecule_.device_view();
    const auto out_dev = device_view();
    const std::size_t num_elems = elements_.num();
    const std::size_t num_atoms = molecule_.num();
    const double eps = params_.phys_eps_;
    const double kappa = params_.phys_kappa_;

    cudaStream_t stream = nullptr;
    double* energy_dev = nullptr;
    auto check = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            std::cerr << "[CUDA_OUTPUT] " << what << " failed: "
                      << cudaGetErrorString(err) << "\n";
            std::exit(1);
        }
    };

    check(cudaMalloc(&energy_dev, sizeof(double)), "cudaMalloc solvation_energy");
    check(cudaMemsetAsync(energy_dev, 0, sizeof(double), stream),
          "cudaMemsetAsync solvation_energy");

    output_solvation_cuda(elem_dev.x, elem_dev.y, elem_dev.z,
                          elem_dev.nx, elem_dev.ny, elem_dev.nz,
                          elem_dev.area,
                          mol_dev.particles_x, mol_dev.particles_y,
                          mol_dev.particles_z, mol_dev.charge,
                          out_dev.potential, potential_offset_,
                          num_elems, num_atoms,
                          eps, kappa,
                          energy_dev, stream);
    CUDA_CHECK_LAST_KERNEL();

    const auto copy_t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(&solvation_energy, energy_dev, sizeof(double),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync solvation_energy");
    const auto copy_t1 = std::chrono::steady_clock::now();
    tabipb_cuda_stats::record_memcpy(
        cudaMemcpyDeviceToHost, sizeof(double),
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(copy_t1 - copy_t0).count()));
    CUDA_STREAM_SYNC_AND_CHECK(stream);
    cudaFree(energy_dev);

    return true;
}
#endif
