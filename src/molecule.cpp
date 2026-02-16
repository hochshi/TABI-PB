#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
// #include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#include "molecule.h"

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#endif

Molecule::Molecule(struct Params &params, struct Timers_Molecule &timers)
    : Particles(params), timers_(timers) {
  timers_.ctor.start();

  std::string line;
  while (std::getline(params.pqr_file_, line)) {

    std::istringstream iss(line);
    std::vector<std::string> tokenized_line{
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>{}};

    if (tokenized_line[0] == "ATOM") {
      x_.push_back(std::stod(tokenized_line[5]));
      y_.push_back(std::stod(tokenized_line[6]));
      z_.push_back(std::stod(tokenized_line[7]));
      charge_.push_back(std::stod(tokenized_line[8]));
      radius_.push_back(std::stod(tokenized_line[9]));
    }
  }

  num_ = radius_.size();
  order_.resize(num_);
  std::iota(order_.begin(), order_.end(), 0);

  timers_.ctor.stop();
}

void Molecule::build_xyzr_file() const {
  timers_.build_xyzr_file.start();

  std::ofstream xyzr_file("molecule.xyzr");

  for (std::size_t i = 0; i < num_; ++i) {
    xyzr_file << x_[i] << " " << y_[i] << " " << z_[i] << " " << radius_[i]
              << std::endl;
  }

  xyzr_file.close();

  timers_.build_xyzr_file.stop();
}

void Molecule::reorder() {
#ifdef USE_CUDA_CC
  if (device_state_ != CudaDeviceState::HostOnly) {
    std::fprintf(stderr,
                 "[CUDA] Molecule::reorder called after device copyin. "
                 "Call delete_from_device() first or avoid reordering.\n");
    std::abort();
  }
#endif
  apply_order(order_.begin(), order_.end(), charge_.begin());
  apply_order(order_.begin(), order_.end(), radius_.begin());
}

void Molecule::unorder() {
#ifdef USE_CUDA_CC
  if (device_state_ != CudaDeviceState::HostOnly) {
    std::fprintf(stderr,
                 "[CUDA] Molecule::unorder called after device copyin. "
                 "Call delete_from_device() first or avoid unordering.\n");
    std::abort();
  }
#endif
  apply_unorder(order_.begin(), order_.end(), x_.begin());
  apply_unorder(order_.begin(), order_.end(), y_.begin());
  apply_unorder(order_.begin(), order_.end(), z_.begin());

  apply_unorder(order_.begin(), order_.end(), charge_.begin());
  apply_unorder(order_.begin(), order_.end(), radius_.begin());
}

void Molecule::copyin_to_device() const {
  timers_.copyin_to_device.start();

#ifdef USE_CUDA_CC
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  const bool require_all =
      (require_all_env && std::strcmp(require_all_env, "0") != 0);

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

  auto &buf = device_buffers_;
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
    CUDA_MEMCPY_ASYNC(buf.particles_x_dev, x_.data(),
                      x_num * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.particles_y_dev, y_.data(),
                      y_num * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.particles_z_dev, z_.data(),
                      z_num * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(buf.charge_dev, charge_.data(),
                      charge_num * sizeof(double), cudaMemcpyHostToDevice,
                      stream);
  }

  CUDA_SYNC_AND_CHECK();
  buf.ready = true;
  device_state_ = CudaDeviceState::DeviceMapped;

  if (require_all && num_particles > 0) {
    if (!buf.particles_x_dev || !buf.particles_y_dev ||
        !buf.particles_z_dev || !buf.charge_dev) {
      std::fprintf(stderr,
                   "[CUDA] Molecule copyin missing device buffers under "
                   "TABIPB_CUDA_REQUIRE_ALL=1\n");
      std::abort();
    }
  }
#endif

  timers_.copyin_to_device.stop();
}

void Molecule::delete_from_device() const {
  timers_.delete_from_device.start();

#ifdef USE_CUDA_CC
  auto &buf = device_buffers_;
  CUDA_FREE_AND_NULL(buf.particles_x_dev);
  CUDA_FREE_AND_NULL(buf.particles_y_dev);
  CUDA_FREE_AND_NULL(buf.particles_z_dev);
  CUDA_FREE_AND_NULL(buf.charge_dev);
  buf.num_particles = 0;
  buf.ready = false;
  device_state_ = CudaDeviceState::HostOnly;
#endif

  timers_.delete_from_device.stop();
}

void Timers_Molecule::print() const {
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(5);
  std::cout << "|...Molecule function times (s)...." << std::endl;
  std::cout << "|   |...ctor.......................: ";
  std::cout << std::setw(12) << std::right << ctor.elapsed_time() << std::endl;
  std::cout << "|   |...build_xyzr_file............: ";
  std::cout << std::setw(12) << std::right << build_xyzr_file.elapsed_time()
            << std::endl;
#ifdef USE_CUDA_CC
  std::cout << "|   |...copyin_to_device...........: ";
  std::cout << std::setw(12) << std::right << copyin_to_device.elapsed_time()
            << std::endl;
  std::cout << "|   |...delete_from_device.........: ";
  std::cout << std::setw(12) << std::right << delete_from_device.elapsed_time()
            << std::endl;
#endif
  std::cout << "|" << std::endl;
}

std::string Timers_Molecule::get_durations() const {
  std::string durations;
  durations.append(std::to_string(ctor.elapsed_time())).append(", ");
  durations.append(std::to_string(build_xyzr_file.elapsed_time())).append(", ");
  durations.append(std::to_string(copyin_to_device.elapsed_time()))
      .append(", ");
  durations.append(std::to_string(delete_from_device.elapsed_time()))
      .append(", ");

  return durations;
}

std::string Timers_Molecule::get_headers() const {
  std::string headers;
  headers.append("Molecule ctor, ");
  headers.append("Molecule build_xyzr_file, ");
  headers.append("Molecule copyin_to_device, ");
  headers.append("Molecule delete_from_device, ");

  return headers;
}
