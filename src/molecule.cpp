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
  copyin_to_device_cuda_();
#endif

  timers_.copyin_to_device.stop();
}

void Molecule::delete_from_device() const {
  timers_.delete_from_device.start();

#ifdef USE_CUDA_CC
  delete_from_device_cuda_();
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
