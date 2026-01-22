cmake_minimum_required(VERSION 3.12)
project(TABI-PB VERSION 3.0 LANGUAGES CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0042 NEW) # MacOS RPATH on by default
cmake_policy(SET CMP0048 NEW) # project() command manages the VERSION variables
cmake_policy(SET CMP0054 NEW) # Only interpret if() arguments as variables or keywords when unquoted
cmake_policy(SET CMP0077 NEW) # option() honors normal variables (i.e. does nothing if a normal variable with the same name exists)
cmake_policy(SET CMP0083 NEW) # Pass flags needed for position-independent executables
cmake_policy(SET CMP0091 NEW) # MSVC runtime library flags are selected by an abstraction (i.e. CMAKE_MSVC_RUNTIME_LIBRARY)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 11)


include(FetchContent)
include(ExternalProject)

################################################################################
# CMake modules
################################################################################
# set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


################################################################################
# Build type
################################################################################
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")

  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)

  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo")
endif ()


################################################################################
# Install directories
################################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)


################################################################################
# OpenACC
################################################################################
option(ENABLE_OPENACC "OpenACC" OFF)

if (ENABLE_OPENACC)
    find_package(OpenACC REQUIRED)
    set(OPENACC_ACCEL_TARGET tesla)
    add_definitions(-DOPENACC_ENABLED)
endif ()


################################################################################
# OpenMP
################################################################################
option(ENABLE_OPENMP "OpenMP" OFF)

if (ENABLE_OPENMP)
    find_package(OpenMP)
    add_definitions(-DOPENMP_ENABLED)
endif ()


################################################################################
# TinyPLY
################################################################################
option(ENABLE_PLY "PLY" OFF)

if(ENABLE_PLY)

  FetchContent_Declare( tinyply
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG 40aa4a0ae9e9c203e11893f78b8bcaf8a50e65f0 # 2.3.4
  )

  FetchContent_GetProperties(tinyply)
  if(NOT tinyply_POPULATED)
    FetchContent_Populate(tinyply)
    add_library(tinyply STATIC
      ${tinyply_SOURCE_DIR}/source/tinyply.cpp
      ${tinyply_SOURCE_DIR}/source/tinyply.h
    )
  include_directories(${tinyply_SOURCE_DIR}/source)
  endif()
  
  add_definitions(-DPLY_ENABLED)
  link_libraries(
    tinyply
  )
endif()

################################################################################
# Getting nanoshaper binary
################################################################################
option(GET_NanoShaper "Put NanoShaper executable in the build directory" OFF)

if (GET_NanoShaper)
    set(MESH_ROOT_URL_PATH "https://github.com/lwwilson1/mesh_routines/releases/download/v1.6")
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

    set(NANOSHAPER_FULL_URL_PATH)
    set(NANOSHAPER_RAW_BINARY_NAME "NanoShaper")
    message(STATUS "Copying NanoShaper executable")

    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")      # Mac OS X specific code
        set(NANOSHAPER_FULL_URL_PATH "${MESH_ROOT_URL_PATH}/NanoShaper_OSX")
    elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")   # Linux specific code
        set(NANOSHAPER_FULL_URL_PATH "${MESH_ROOT_URL_PATH}/NanoShaper_Linux64")
    elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows") # Windows specific code
        set(NANOSHAPER_FULL_URL_PATH "${MESH_ROOT_URL_PATH}/NanoShaper32.exe")
        set(NANOSHAPER_RAW_BINARY_NAME "NanoShaper.exe")
    endif ()

    if (NOT EXISTS "${EXECUTABLE_OUTPUT_PATH}/${NANOSHAPER_RAW_BINARY_NAME}")
        file(DOWNLOAD ${NANOSHAPER_FULL_URL_PATH} ${EXECUTABLE_OUTPUT_PATH}/${NANOSHAPER_RAW_BINARY_NAME})
        message(STATUS "NanoShaper download: ${NANOSHAPER_FULL_URL_PATH} ${EXECUTABLE_OUTPUT_PATH}/${NANOSHAPER_RAW_BINARY_NAME}")
    endif ()

    install(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${NANOSHAPER_RAW_BINARY_NAME} DESTINATION bin)
endif ()


################################################################################
# Setting up src builds
################################################################################
add_subdirectory(src)
