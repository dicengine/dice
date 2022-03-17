cmake_minimum_required(VERSION 3.4)
MESSAGE("\nConfiguring DICe::dev\n")

PROJECT(DICe::dev)

IF(NOT APPLE)
MESSAGE(FATAL_ERROR "Not implemented for Windows yet")
ENDIF()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Using LAPACK from: ${LAPACK}")
MESSAGE(STATUS "Using BLAS from: ${BLAS}")
MESSAGE(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Using Fortran compiler: ${CMAKE_Fortran_COMPILER}")
MESSAGE(STATUS "Install directory: ${CMAKE_INSTALL_PREFIX}")

# Make sure that BLAS and LAPACK are installed

if(NOT DEFINED BLAS)
  if(EXISTS "/opt/intel/oneapi/mkl/2022.0.0/lib/libmkl_blas95_ilp64.a")
     MESSAGE(STATUS "Found BLAS in /opt/intel/oneapi/mkl")
  else()
     MESSAGE(FATAL_ERROR "Could not find BLAS, please install before proceeding")
  endif()
endif()
if(NOT DEFINED LAPACK)
  if(EXISTS "/opt/intel/oneapi/mkl/2022.0.0/lib/libmkl_lapack95_ilp64.a")
     MESSAGE(STATUS "Found LAPACK in /opt/intel/oneapi/mkl")
  else()
     MESSAGE(FATAL_ERROR "Could not find LAPACK, please install before proceeding")
  endif()
endif()

include(ExternalProject)

# install dev-pack (Trilinos netcdf and hdf5 OpenCV)

SET(DEV_PACK_CMAKE_ARGS -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -D CMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER} -D CMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER} -D CMAKE_Fortran_COMPILER:FILEPATH=${CMAKE_Fortran_COMPILER} -D LAPACK:STRING=${LAPACK} -D BLAS:STRING=${BLAS} -D OPENCV_GIT_TAG:STRING=dad2633)

ExternalProject_Add(dev-pack
  GIT_REPOSITORY git@github.com:dicengine/dice-dev-pack-mac.git
  GIT_TAG main
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/tmp/dev-pack
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dev-pack
  INSTALL_COMMAND ""
  CMAKE_ARGS ${DEV_PACK_CMAKE_ARGS}
)

# install DICe::GUI

ExternalProject_Add(gui
  GIT_REPOSITORY git@github.com:dicengine/dicegui.git
  PREFIX       ${CMAKE_CURRENT_BINARY_DIR}/tmp/gui
  SOURCE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/gui
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
)

# install DICe

IF(NOT DEFINED DICE_DEBUG_MSG)
 SET(DICE_DEBUG_MSG OFF)
ENDIF()

SET(DICE_COMPILER_FLAGS "-Wall -Wno-unused-local-typedefs -Wno-user-defined-warnings -Wsign-compare -Wunused-variable -Wno-deprecated-declarations")
SET(DICE_CMAKE_ARGS -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -D CMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX} -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF -D DICE_DEBUG_MSG:BOOL=${DICE_DEBUG_MSG} -D DICE_USE_DOUBLE:BOOL=ON -D DICE_ENABLE_GLOBAL:BOOL=ON -D NetCDF_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/dev-pack/netcdf/lib -D HDF5_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/dev-pack/hdf5/lib -D OpenCV_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/dev-pack/opencv -D CMAKE_CXX_FLAGS:STRING=${DICE_COMPILER_FLAGS} -D DICE_TRILINOS_DIR=${CMAKE_CURRENT_BINARY_DIR}/dev-pack/trilinos)

ExternalProject_Add(dice
  DOWNLOAD_COMMAND ""
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/tmp/dice
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../
  CMAKE_ARGS ${DICE_CMAKE_ARGS}
)
add_dependencies(dice dev-pack)

# force re-configure of dice
ExternalProject_Add_Step(dice forceconfigure
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/clean-dice-cmake.cmake
  DEPENDERS configure
ALWAYS 1)
