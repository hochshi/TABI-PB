# CXX code for standalone
add_executable(tabipb main.cpp
        main_backend.h
        main_backend_cpu.cpp
        main_backend_cuda.cpp
        params.cpp params.h
        particles.cpp particles.h
        molecule.cpp molecule.h
        molecule_backend_cuda.cpp
        elements.cpp elements.h
        elements_backend_common.h
        elements_backend_cpu.h elements_backend_cpu.cpp
        elements_backend_cuda.h elements_backend_cuda.cpp
        tree.cpp tree.h
        interp_pts.cpp interp_pts.h
        interp_pts_backend_cpu.h interp_pts_backend_cpu.cpp
        interp_pts_backend_cuda.cpp
        interaction_list.cpp interaction_list.h
        tree_compute.h
        coulombic_energy_compute.cpp coulombic_energy_compute.h
        coulombic_backend_cuda.cpp
        solvation_energy_compute.cpp solvation_energy_compute.h
        solvation_backend_cuda.cpp
        source_term_compute.cpp source_term_compute.h
        source_term_backend_cuda.cpp
        boundary_element.cpp gmres.cpp
        precondition.cpp boundary_element.h
        output.cpp output.h output_backend_cuda.cpp
        tabipb_timers.h timer.h constants.h)

if (ENABLE_CUDA_CC)
    target_sources(tabipb PRIVATE cc_cuda.cu cc_cuda.h up_cuda.cu up_cuda.h source_term_cuda.cu source_term_cuda.h coulombic_energy_cuda.cu coulombic_energy_cuda.h solvation_energy_cuda.cu solvation_energy_cuda.h be_cuda.cu be_cuda.h elements_cuda.cu elements_cuda.h down_cuda.cu down_cuda.h pc_cuda.cu pc_cuda.h pp_cuda.cu pp_cuda.h pppc_cuda.cu pppc_cuda.h gmres_cuda.cu gmres_cuda.h precondition_cuda.cu precondition_cuda.h interp_pts_cuda.cu interp_pts_cuda.h output_cuda.cu output_cuda.h)
    target_compile_definitions(tabipb PRIVATE USE_CUDA_CC)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # Avoid forwarding NVHPC-specific host flags (e.g. -gpu=ccXX) into nvcc.
        set_target_properties(tabipb PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif ()
    set_source_files_properties(cc_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(up_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(source_term_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(coulombic_energy_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(solvation_energy_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(be_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(elements_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(down_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(pc_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(pp_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(pppc_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(gmres_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(precondition_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(interp_pts_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(output_cuda.cu PROPERTIES CUDA_STANDARD 11)
    target_link_libraries(tabipb PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
endif ()

target_compile_features(tabipb PRIVATE cxx_std_11)
target_compile_options(tabipb PRIVATE 
                       $<$<CONFIG:RELEASE>:-O3>
                       $<$<CONFIG:RELWITHDEBINFO>:-O3>
                       $<$<CONFIG:DEBUG>:-O0 -Wall>)

if (ENABLE_OPENMP)
    target_link_libraries(tabipb PRIVATE OpenMP::OpenMP_CXX)
endif ()

#Math linking is unnecessary for Windows
if (NOT WIN32)
    target_link_libraries(tabipb PRIVATE m)
endif ()

install (TARGETS tabipb DESTINATION bin)



################################################
###### For APBS 
################################################

#option(ENABLE_TABIPB_APBS "TABIPB APBS" OFF)
if (ENABLE_TABIPB_APBS)
    add_definitions(-DTABIPB_APBS)
    include_directories(
        ../../../src/
        ../../../build/src/
        ../../../externals/fetk/maloc/src/base
        ../../../externals/fetk/maloc/src/vsys
        ../../../externals/fetk/maloc/src/vsh
        ../../../externals/fetk/maloc/src/psh)
  
    option(BUILD_SHARED_LIBS "Build shared libraries." OFF)
  
    set(LIBFILES
        main_backend.h
        main_backend_cpu.cpp
        main_backend_cuda.cpp
        params.cpp params.h particles.cpp particles.h 
        molecule.cpp molecule.h molecule_backend_cuda.cpp elements.cpp elements.h tree.cpp tree.h
        interp_pts.cpp interp_pts.h interp_pts_backend_cpu.h interp_pts_backend_cpu.cpp interp_pts_backend_cuda.cpp
        elements_backend_common.h
        elements_backend_cpu.h elements_backend_cpu.cpp
        elements_backend_cuda.h elements_backend_cuda.cpp
        interaction_list.cpp interaction_list.h tree_compute.h
        coulombic_energy_compute.cpp coulombic_energy_compute.h
        coulombic_backend_cuda.cpp
        solvation_energy_compute.cpp solvation_energy_compute.h
        solvation_backend_cuda.cpp
        source_term_compute.cpp source_term_compute.h
        source_term_backend_cuda.cpp
        boundary_element.cpp gmres.cpp precondition.cpp 
        boundary_element.h constants.h
        output.cpp output.h output_backend_cuda.cpp tabipb_timers.h timer.h
        tabipb_wrap/TABIPBWrap.cpp tabipb_wrap/TABIPBWrap.h
        tabipb_wrap/TABIPBStruct.h tabipb_wrap/params_apbs_ctor.cpp
        tabipb_wrap/molecule_apbs_ctor.cpp)

    if (ENABLE_CUDA_CC)
        list(APPEND LIBFILES cc_cuda.cu cc_cuda.h up_cuda.cu up_cuda.h source_term_cuda.cu source_term_cuda.h coulombic_energy_cuda.cu coulombic_energy_cuda.h solvation_energy_cuda.cu solvation_energy_cuda.h be_cuda.cu be_cuda.h elements_cuda.cu elements_cuda.h down_cuda.cu down_cuda.h pc_cuda.cu pc_cuda.h pp_cuda.cu pp_cuda.h pppc_cuda.cu pppc_cuda.h gmres_cuda.cu gmres_cuda.h precondition_cuda.cu precondition_cuda.h interp_pts_cuda.cu interp_pts_cuda.h output_cuda.cu output_cuda.h)
    endif ()
  
    set(TABIPB_LIBNAME TABIPBlib)
    add_library(${TABIPB_LIBNAME} ${LIBFILES})
    target_compile_features(${TABIPB_LIBNAME} PRIVATE cxx_std_11)
    target_compile_options(${TABIPB_LIBNAME} PRIVATE 
                           $<$<CONFIG:RELEASE>:-O3>
                           $<$<CONFIG:RELWITHDEBINFO>:-O3>
                           $<$<CONFIG:DEBUG>:-O0 -Wall>)

    if (ENABLE_CUDA_CC)
        target_compile_definitions(${TABIPB_LIBNAME} PRIVATE USE_CUDA_CC)
        if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            # Avoid forwarding NVHPC-specific host flags (e.g. -gpu=ccXX) into nvcc.
            set_target_properties(${TABIPB_LIBNAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
        endif ()
        set_source_files_properties(cc_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(up_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(source_term_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(coulombic_energy_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(solvation_energy_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(be_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(elements_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(down_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(pc_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(pp_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(pppc_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(gmres_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(precondition_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(interp_pts_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(output_cuda.cu PROPERTIES CUDA_STANDARD 11)
        target_link_libraries(${TABIPB_LIBNAME} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
    endif ()

endif ()
