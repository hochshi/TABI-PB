#ifndef H_TABIPB_MAIN_BACKEND_H
#define H_TABIPB_MAIN_BACKEND_H

inline const char* main_backend_name() {
#ifdef USE_CUDA_CC
  return "CUDA";
#elif defined(OPENACC_ENABLED) || defined(_OPENACC)
  return "OpenACC";
#else
  return "CPU";
#endif
}

void main_backend_initialize_runtime();

#endif
