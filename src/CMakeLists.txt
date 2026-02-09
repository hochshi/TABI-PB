# CXX code for standalone
add_executable(tabipb main.cpp
        params.cpp params.h
        particles.cpp particles.h
        molecule.cpp molecule.h
        elements.cpp elements.h
        tree.cpp tree.h
        interp_pts.cpp interp_pts.h
        interaction_list.cpp interaction_list.h
        tree_compute.h
        coulombic_energy_compute.cpp coulombic_energy_compute.h
        solvation_energy_compute.cpp solvation_energy_compute.h
        source_term_compute.cpp source_term_compute.h
        boundary_element.cpp gmres.cpp
        precondition.cpp boundary_element.h
        output.cpp output.h
        tabipb_timers.h timer.h constants.h)

if (ENABLE_CUDA_CC)
    target_sources(tabipb PRIVATE cc_cuda.cu cc_cuda.h up_cuda.cu up_cuda.h)
    target_compile_definitions(tabipb PRIVATE USE_CUDA_CC)
    set_source_files_properties(cc_cuda.cu PROPERTIES CUDA_STANDARD 11)
    set_source_files_properties(up_cuda.cu PROPERTIES CUDA_STANDARD 11)
    target_link_libraries(tabipb PRIVATE CUDA::cudart CUDA::cuda_driver)
endif ()

target_compile_features(tabipb PRIVATE cxx_std_11)
target_compile_options(tabipb PRIVATE 
                       $<$<CONFIG:RELEASE>:-O3>
                       $<$<CONFIG:RELWITHDEBINFO>:-O3>
                       $<$<CONFIG:DEBUG>:-O0 -Wall>)

if (ENABLE_OPENACC)
    target_link_libraries(tabipb PRIVATE OpenACC::OpenACC_CXX -acc)
    target_compile_definitions(tabipb PRIVATE OPENACC_ENABLED)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        target_compile_options(tabipb PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:NVHPC>>:-Minfo=accel>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:NVHPC>>:-Minfo=accel>
        )
        # target_compile_options(tabipb PRIVATE -Minfo=accel)
        if (CMAKE_CUDA_ARCHITECTURES)
            set(_tabipb_gpu_list "")
            foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
                if (_arch MATCHES "^cc")
                    list(APPEND _tabipb_gpu_list "${_arch}")
                else ()
                    list(APPEND _tabipb_gpu_list "cc${_arch}")
                endif ()
            endforeach ()
            list(JOIN _tabipb_gpu_list "," _tabipb_gpu_list_str)
            if (_tabipb_gpu_list_str)
                target_compile_options(tabipb PRIVATE "-gpu=${_tabipb_gpu_list_str}")
            endif ()
        endif ()
    endif ()
endif ()

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
        params.cpp params.h particles.cpp particles.h 
        molecule.cpp molecule.h elements.cpp elements.h tree.cpp tree.h
        interaction_list.cpp interaction_list.h tree_compute.h
        coulombic_energy_compute.cpp coulombic_energy_compute.h
        solvation_energy_compute.cpp solvation_energy_compute.h
        source_term_compute.cpp source_term_compute.h
        boundary_element.cpp gmres.cpp precondition.cpp 
        boundary_element.h constants.h
        output.cpp output.h tabipb_timers.h timer.h
        tabipb_wrap/TABIPBWrap.cpp tabipb_wrap/TABIPBWrap.h
        tabipb_wrap/TABIPBStruct.h tabipb_wrap/params_apbs_ctor.cpp
        tabipb_wrap/molecule_apbs_ctor.cpp)

    if (ENABLE_CUDA_CC)
        list(APPEND LIBFILES cc_cuda.cu cc_cuda.h up_cuda.cu up_cuda.h)
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
        set_source_files_properties(cc_cuda.cu PROPERTIES CUDA_STANDARD 11)
        set_source_files_properties(up_cuda.cu PROPERTIES CUDA_STANDARD 11)
        target_link_libraries(${TABIPB_LIBNAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif ()

    if (ENABLE_OPENACC)
        target_link_libraries(${TABIPB_LIBNAME} PRIVATE OpenACC::OpenACC_CXX -acc)
        target_compile_definitions(${TABIPB_LIBNAME} PRIVATE OPENACC_ENABLED)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
            target_compile_options(${TABIPB_LIBNAME} PRIVATE -Minfo=accel)
            if (CMAKE_CUDA_ARCHITECTURES)
                set(_tabipb_gpu_list "")
                foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
                    if (_arch MATCHES "^cc")
                        list(APPEND _tabipb_gpu_list "${_arch}")
                    else ()
                        list(APPEND _tabipb_gpu_list "cc${_arch}")
                    endif ()
                endforeach ()
                list(JOIN _tabipb_gpu_list "," _tabipb_gpu_list_str)
                if (_tabipb_gpu_list_str)
                    target_compile_options(${TABIPB_LIBNAME} PRIVATE "-gpu=${_tabipb_gpu_list_str}")
                endif ()
            endif ()
        endif ()
    endif ()

endif ()
