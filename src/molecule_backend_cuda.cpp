#include "molecule.h"

#ifdef USE_CUDA_CC
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_helpers.h"

void Molecule::copyin_to_device_cuda_() const {
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  const bool require_all =
      !(require_all_env && std::strcmp(require_all_env, "0") == 0);

  const std::size_t num_particles = num_;
  const std::size_t x_num = x_.size();
  const std::size_t y_num = y_.size();
  const std::size_t z_num = z_.size();
  const std::size_t charge_num = charge_.size();

  if (x_num != num_particles || y_num != num_particles ||
      z_num != num_particles || charge_num != num_particles) {
    std::fprintf(stderr,
                 "[CUDA] Molecule size mismatch: num=%zu x=%zu y=%zu z=%zu "
                 "charge=%zu\n",
                 num_particles, x_num, y_num, z_num, charge_num);
    std::abort();
  }

  auto& buf = device_buffers_;
  if (buf.num_particles != 0 && buf.num_particles != num_particles) {
    CUDA_FREE_AND_NULL(buf.particles_x_dev);
    CUDA_FREE_AND_NULL(buf.particles_y_dev);
    CUDA_FREE_AND_NULL(buf.particles_z_dev);
    CUDA_FREE_AND_NULL(buf.charge_dev);
    buf.num_particles = 0;
    buf.ready = false;
  }

  if (buf.num_particles == 0 && num_particles > 0) {
    CUDA_MALLOC_OR_DIE(&buf.particles_x_dev, x_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&buf.particles_y_dev, y_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&buf.particles_z_dev, z_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&buf.charge_dev, charge_num * sizeof(double));
    buf.num_particles = num_particles;
  }

  cudaStream_t stream = nullptr;

  if (num_particles > 0) {
    CUDA_MEMCPY_ASYNC(buf.particles_x_dev, x_.data(), x_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.particles_y_dev, y_.data(), y_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.particles_z_dev, z_.data(), z_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.charge_dev, charge_.data(), charge_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
  }

  CUDA_SYNC_AND_CHECK();
  buf.ready = true;
  device_state_ = CudaDeviceState::DeviceMapped;

  if (require_all && num_particles > 0) {
    if (!buf.particles_x_dev || !buf.particles_y_dev || !buf.particles_z_dev ||
        !buf.charge_dev) {
      std::fprintf(stderr,
                   "[CUDA] Molecule copyin missing device buffers under "
                   "TABIPB_CUDA_REQUIRE_ALL=1\n");
      std::abort();
    }
  }
}

void Molecule::delete_from_device_cuda_() const {
  auto& buf = device_buffers_;
  CUDA_FREE_AND_NULL(buf.particles_x_dev);
  CUDA_FREE_AND_NULL(buf.particles_y_dev);
  CUDA_FREE_AND_NULL(buf.particles_z_dev);
  CUDA_FREE_AND_NULL(buf.charge_dev);
  buf.num_particles = 0;
  buf.ready = false;
  device_state_ = CudaDeviceState::HostOnly;
}
#endif
