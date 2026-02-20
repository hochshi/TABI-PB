#include "elements_backend_cuda.h"

#ifdef USE_CUDA_CC
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_helpers.h"
#include "elements_cuda.h"
#endif

bool elements_try_compute_source_term_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    std::size_t num_elements,
    std::size_t num_atoms,
    double eps_solute,
    void* stream) {
#ifdef USE_CUDA_CC
  const bool view_ok = (elem_view.num == num_elements) &&
                       (mol_view.num_particles == num_atoms) && elem_view.x &&
                       elem_view.y && elem_view.z && elem_view.nx &&
                       elem_view.ny && elem_view.nz && elem_view.source_term &&
                       mol_view.particles_x && mol_view.particles_y &&
                       mol_view.particles_z && mol_view.charge;
  if (!view_ok) {
    return false;
  }

  elements_compute_source_term_cuda(
      elem_view.x, elem_view.y, elem_view.z, elem_view.nx, elem_view.ny,
      elem_view.nz, mol_view.particles_x, mol_view.particles_y,
      mol_view.particles_z, mol_view.charge, elem_view.source_term, num_elements,
      num_atoms, eps_solute, stream);
  return true;
#else
  (void)elem_view;
  (void)mol_view;
  (void)num_elements;
  (void)num_atoms;
  (void)eps_solute;
  (void)stream;
  return false;
#endif
}

bool elements_try_compute_charges_cuda(
    const Elements::View& elem_view,
    const double* potential_dev,
    std::size_t num,
    void* stream) {
#ifdef USE_CUDA_CC
  if (!cuda_pointer_is_device_accessible(potential_dev)) {
    return false;
  }

  const bool view_ok = (elem_view.num == num) && elem_view.nx && elem_view.ny &&
                       elem_view.nz && elem_view.area && elem_view.target_q &&
                       elem_view.target_q_dx && elem_view.target_q_dy &&
                       elem_view.target_q_dz && elem_view.source_q &&
                       elem_view.source_q_dx && elem_view.source_q_dy &&
                       elem_view.source_q_dz;
  if (!view_ok) {
    return false;
  }

  elements_compute_charges_cuda(
      elem_view.nx, elem_view.ny, elem_view.nz, elem_view.area, potential_dev,
      elem_view.target_q, elem_view.target_q_dx, elem_view.target_q_dy,
      elem_view.target_q_dz, elem_view.source_q, elem_view.source_q_dx,
      elem_view.source_q_dy, elem_view.source_q_dz, num, stream);
  return true;
#else
  (void)elem_view;
  (void)potential_dev;
  (void)num;
  (void)stream;
  return false;
#endif
}

#ifdef USE_CUDA_CC
void Elements::copyin_to_device_cuda_() const {
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  const bool require_all =
      (require_all_env && std::strcmp(require_all_env, "0") != 0);
  const std::size_t num = num_;
  const std::size_t x_num = x_.size();
  const std::size_t y_num = y_.size();
  const std::size_t z_num = z_.size();
  const std::size_t nx_num = nx_.size();
  const std::size_t ny_num = ny_.size();
  const std::size_t nz_num = nz_.size();
  const std::size_t area_num = area_.size();
  const std::size_t source_term_num = source_term_.size();
  const std::size_t tq_num = target_charge_.size();
  const std::size_t tq_dx_num = target_charge_dx_.size();
  const std::size_t tq_dy_num = target_charge_dy_.size();
  const std::size_t tq_dz_num = target_charge_dz_.size();
  const std::size_t sq_num = source_charge_.size();
  const std::size_t sq_dx_num = source_charge_dx_.size();
  const std::size_t sq_dy_num = source_charge_dy_.size();
  const std::size_t sq_dz_num = source_charge_dz_.size();

  if (x_num != num || y_num != num || z_num != num || nx_num != num ||
      ny_num != num || nz_num != num || area_num != num || tq_num != num ||
      tq_dx_num != num || tq_dy_num != num || tq_dz_num != num ||
      sq_num != num || sq_dx_num != num || sq_dy_num != num || sq_dz_num != num ||
      source_term_num != 2 * num) {
    std::fprintf(stderr,
                 "[CUDA_ELEMENTS] size mismatch: num=%zu x=%zu y=%zu z=%zu "
                 "nx=%zu ny=%zu nz=%zu area=%zu source_term=%zu "
                 "tq=%zu tq_dx=%zu tq_dy=%zu tq_dz=%zu "
                 "sq=%zu sq_dx=%zu sq_dy=%zu sq_dz=%zu\n",
                 num, x_num, y_num, z_num, nx_num, ny_num, nz_num, area_num,
                 source_term_num, tq_num, tq_dx_num, tq_dy_num, tq_dz_num,
                 sq_num, sq_dx_num, sq_dy_num, sq_dz_num);
    std::abort();
  }

  auto& ptrs = device_buffers_;
  if (ptrs.ready && ptrs.num != num) {
    CUDA_FREE_AND_NULL(ptrs.x);
    CUDA_FREE_AND_NULL(ptrs.y);
    CUDA_FREE_AND_NULL(ptrs.z);
    CUDA_FREE_AND_NULL(ptrs.nx);
    CUDA_FREE_AND_NULL(ptrs.ny);
    CUDA_FREE_AND_NULL(ptrs.nz);
    CUDA_FREE_AND_NULL(ptrs.area);
    CUDA_FREE_AND_NULL(ptrs.source_term);
    CUDA_FREE_AND_NULL(ptrs.target_q);
    CUDA_FREE_AND_NULL(ptrs.target_q_dx);
    CUDA_FREE_AND_NULL(ptrs.target_q_dy);
    CUDA_FREE_AND_NULL(ptrs.target_q_dz);
    CUDA_FREE_AND_NULL(ptrs.source_q);
    CUDA_FREE_AND_NULL(ptrs.source_q_dx);
    CUDA_FREE_AND_NULL(ptrs.source_q_dy);
    CUDA_FREE_AND_NULL(ptrs.source_q_dz);
    reset_device_buffers_();
  }

  if (!ptrs.ready && num > 0) {
    CUDA_MALLOC_OR_DIE(&ptrs.x, x_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.y, y_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.z, z_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.nx, nx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.ny, ny_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.nz, nz_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.area, area_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_term, source_term_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q, tq_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dx, tq_dx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dy, tq_dy_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dz, tq_dz_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q, sq_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dx, sq_dx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dy, sq_dy_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dz, sq_dz_num * sizeof(double));
    ptrs.num = num;
    ptrs.ready = true;
  }

  cudaStream_t stream = nullptr;

  if (num > 0) {
    CUDA_MEMCPY_ASYNC(ptrs.x, x_.data(), x_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.y, y_.data(), y_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.z, z_.data(), z_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.nx, nx_.data(), nx_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.ny, ny_.data(), ny_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.nz, nz_.data(), nz_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.area, area_.data(), area_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.source_term, source_term_.data(),
                      source_term_num * sizeof(double), cudaMemcpyHostToDevice,
                      stream);
  }

  CUDA_SYNC_AND_CHECK();
  device_state_ = CudaDeviceState::DeviceMapped;

  if (require_all && num > 0) {
    if (!ptrs.ready || !ptrs.x || !ptrs.y || !ptrs.z || !ptrs.nx || !ptrs.ny ||
        !ptrs.nz || !ptrs.area || !ptrs.source_term || !ptrs.target_q ||
        !ptrs.target_q_dx || !ptrs.target_q_dy || !ptrs.target_q_dz ||
        !ptrs.source_q || !ptrs.source_q_dx || !ptrs.source_q_dy ||
        !ptrs.source_q_dz) {
      std::fprintf(stderr,
                   "[CUDA_ELEMENTS] copyin missing device buffers under "
                   "TABIPB_CUDA_REQUIRE_ALL=1\n");
      std::abort();
    }
  }
}

void Elements::update_source_term_on_host_cuda_() {
  const std::size_t source_term_num = source_term_.size();
  if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready ||
      source_term_num == 0 || !device_buffers_.source_term) {
    return;
  }
  cudaStream_t stream = nullptr;
  CUDA_MEMCPY_ASYNC(source_term_.data(), device_buffers_.source_term,
                    source_term_num * sizeof(double), cudaMemcpyDeviceToHost,
                    stream);
  CUDA_STREAM_SYNC_AND_CHECK(stream);
}

void Elements::delete_from_device_cuda_() const {
  if (device_buffers_.ready) {
    CUDA_FREE_AND_NULL(device_buffers_.x);
    CUDA_FREE_AND_NULL(device_buffers_.y);
    CUDA_FREE_AND_NULL(device_buffers_.z);
    CUDA_FREE_AND_NULL(device_buffers_.nx);
    CUDA_FREE_AND_NULL(device_buffers_.ny);
    CUDA_FREE_AND_NULL(device_buffers_.nz);
    CUDA_FREE_AND_NULL(device_buffers_.area);
    CUDA_FREE_AND_NULL(device_buffers_.source_term);
    CUDA_FREE_AND_NULL(device_buffers_.target_q);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dx);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dy);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dz);
    CUDA_FREE_AND_NULL(device_buffers_.source_q);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dx);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dy);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dz);
    reset_device_buffers_();
  }
  device_state_ = CudaDeviceState::HostOnly;
}
#endif
