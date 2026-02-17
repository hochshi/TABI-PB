#ifndef H_TABIPB_ELEMENTS_BACKEND_COMMON_H
#define H_TABIPB_ELEMENTS_BACKEND_COMMON_H

#include <cstdlib>
#include <cstring>

inline bool elements_cuda_require_all() {
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  return (require_all_env && std::strcmp(require_all_env, "0") != 0);
}

#endif
