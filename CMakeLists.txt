cmake_minimum_required(VERSION 3.4)
MESSAGE("\nConfiguring DICe...\n")

MESSAGE(STATUS "Using Trilinos installed in: ${DICE_TRILINOS_DIR}")

# If this is a windows build CLAPACK is required. Find package for
# clapack is automatically enabled by setting the CLAPACK_DIR variable
IF(WIN32)
   # must defined clapack_dir
  IF (NOT DEFINED CLAPACK_DIR)
    MESSAGE(FATAL_ERROR "\nDICe Error: this is a windows build, so cmake must define CLAPACK_DIR:
    (-D CLAPACK_DIR:FILEPATH=<clapack_install_prefix>)!")
  ENDIF()
  FIND_PACKAGE(clapack PATHS ${CLAPACK_DIR})
  IF(clapack_FOUND)
    MESSAGE("\nFound CLAPACK in ${CLAPACK_DIR}")
    link_directories(${CLAPACK_DIR}/F2CLIBS/libf2c)
  ELSE(clapack_FOUND)
    MESSAGE("\nERROR: could not find CLAPACK in the following directory: ${CLAPACK_DIR}")
  ENDIF(clapack_FOUND)
  IF(NOT BUILD_SHARED_LIBS)
    MESSAGE("\nWarning: This is a windows build, but BUILD_SHARED_LIBS is OFF. Setting BUILD_SHARED_LIBS on automatically.")
    SET(BUILD_SHARED_LIBS ON)
  ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(WIN32)

IF(BUILD_SHARED_LIBS)
  MESSAGE(STATUS "BUILD_SHARED_LIBS is set to ON")
ELSE(BUILD_SHARED_LIBS)
  MESSAGE(STATUS "BUILD_SHARED_LIBS is set to OFF")
ENDIF(BUILD_SHARED_LIBS)

# Get Trilinos as one entity
# set(TRILINOS_DIR CACHE PATH "Path to Trilinos directory")
SET(CMAKE_PREFIX_PATH ${DICE_TRILINOS_DIR} ${CMAKE_PREFIX_PATH})
FIND_PACKAGE(Trilinos PATHS ${DICE_TRILINOS_DIR}/lib/cmake/Trilinos)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("End of Trilinos details\n")

SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_VERBOSE_MAKEFILE OFF)

IF(NOT DEFINED DICE_MPI_EXEC)
  SET(DICE_MPI_EXEC mpiexec)
ENDIF()
# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER opton

PROJECT(DICe)

MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
SET(DICE_OUTPUT_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
MESSAGE(STATUS "The output directory for DICe libraries will be: ${DICE_OUTPUT_PREFIX}/lib")
MESSAGE(STATUS "If 'make install' is exectued, the libraries will also be copied to: ${CMAKE_INSTALL_PREFIX}/lib")

# FIND NETCDF
set(DICE_ENABLE_NETCDF OFF)
if(DEFINED NetCDF_DIR)
  set(DICE_ENABLE_NETCDF ON)
  MESSAGE(STATUS "Looking for NetCDF in: ${NetCDF_DIR}")
  find_library(NetCDF_lib NAMES libnetcdf.a netcdf PATHS ${NetCDF_DIR})
  find_library(HDF5_lib NAMES libhdf5.a hdf5 PATHS ${HDF5_DIR})  
  find_library(HDF5_hl_lib NAMES libhdf5_hl.a hdf5_hl PATHS ${HDF5_DIR})  
  MESSAGE(STATUS "Using NetCDF lib: ${NetCDF_lib}")
  MESSAGE(STATUS "Using HDF5 libs: ${HDF5_lib} ${HDF5_hl_lib}")
  IF(NOT NetCDF_lib_NOTFOUND)
      SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${NetCDF_lib})
      SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${HDF5_lib})
      SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${HDF5_hl_lib})
      ADD_DEFINITIONS(-DDICE_ENABLE_NETCDF=1)
      include_directories(${NetCDF_DIR}/../include)
      IF(WIN32)
        FILE(GLOB NetCDF_DLLS ${NetCDF_DIR}/../bin/*.dll)
        FILE(COPY ${NetCDF_DLLS} DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
      ENDIF()
  ELSE()
    message(FATAL_ERROR "Error, NetCDF enabled but not found")
  ENDIF()
else()
  MESSAGE(STATUS "NetCDF will NOT be enabled")
endif()

include(ExternalProject)
MESSAGE(STATUS "Building DICe_utils from ${CMAKE_CURRENT_SOURCE_DIR}")
ExternalProject_Add(DICe_utils
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils_build
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils_tmp
  CMAKE_CACHE_ARGS -DDICE_TRILINOS_DIR:STRING=${DICE_TRILINOS_DIR} -DCMAKE_INSTALL_PREFIX:FILEPATH=${CMAKE_INSTALL_PREFIX} -DBOOST_ROOT:STRING=${BOOST_ROOT} -DBOOST_LIBRARYDIR:STRING=${BOOST_LIBRARYDIR} -DBOOST_INCLUDEDIR:STRING=${BOOST_INCLUDEDIR} -DDICE_ENABLE_NETCDF:STRING=${DICE_ENABLE_NETCDF} -DNetCDF_DIR:STRING=${NetCDF_DIR} -DTIFF_DIR:STRING=${TIFF_DIR} -DJPEG_DIR:STRING=${JPEG_DIR} -DPNG_DIR:STRING=${PNG_DIR} -DDICE_DEBUG_MSG:BOOL=${DICE_DEBUG_MSG} -DCLAPACK_DIR:FILEPATH=${CLAPACK_DIR} -DDICE_USE_DOUBLE:BOOL=${DICE_USE_DOUBLE} -DDICE_OUTPUT_PREFIX:FILEPATH=${DICE_OUTPUT_PREFIX} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  )

# base data type:
if(DICE_USE_DOUBLE)
  ADD_DEFINITIONS(-DDICE_USE_DOUBLE=1)
  MESSAGE(STATUS "Image intensity and scalar type will be: DOUBLE")
else()
  MESSAGE(STATUS "Image intensity and scalar type will be: FLOAT (default)")
endif()

# MPI check -- defaults to TRUE
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(DICE_MPI TRUE)
  ADD_DEFINITIONS(-DDICE_MPI=1)
  MESSAGE(STATUS "Using DICE_MPI_EXEC: ${DICE_MPI_EXEC}")
ELSE()
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(DICE_MPI FALSE)
ENDIF()

# find OpenCV
message(STATUS "Looking for OpenCV in dir: ${OpenCV_DIR}")
find_package( OpenCV NO_DEFAULT_PATH PATHS ${OpenCV_DIR} )
IF(OpenCV_FOUND)
  set(DICE_ENABLE_OPENCV ON)
  IF(WIN32)
    FILE(GLOB OPENCV_DLLS ${OpenCV_DIR}/x64/vc12/bin/*.dll)
    FILE(COPY ${OPENCV_DLLS} DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  ENDIF()
ELSE()
  message(STATUS "OpenCV not found")
ENDIF()


SET(DICE_TRILINOS_HEADERS
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
  )

SET(DICE_TRILINOS_LIB_DIRS
    ${Trilinos_LIBRARY_DIRS}
    ${Trilinos_TPL_LIBRARY_DIRS}
  )

link_directories(${DICE_TRILINOS_LIB_DIRS}
  ${TIFF_DIR}/lib
  ${JPEG_DIR}/lib
  ${PNG_DIR}/lib
  ${CMAKE_INSTALL_PREFIX}/lib)

SET(DICE_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/api
  ${CMAKE_CURRENT_SOURCE_DIR}/src/base
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cine
  ${CMAKE_CURRENT_SOURCE_DIR}/src/netcdf
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fft
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kdtree
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh
  ${TIFF_DIR}
  ${JPEG_DIR}
  ${PNG_DIR}
  ${TIFF_DIR}/include
  ${JPEG_DIR}/include
  ${PNG_DIR}/include
  )

IF(DICE_ENABLE_GLOBAL)
  SET(DICE_HEADER_DIRS
      ${DICE_HEADER_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/global
      ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh/io
      ${CMAKE_CURRENT_SOURCE_DIR}/src/global/triangle)
      add_definitions(-DTRILIBRARY -DANSI_DECLARATORS -DDICE_ENABLE_GLOBAL=1)
ENDIF()

IF(DICE_ENABLE_OPENCV)
  SET(DICE_HEADER_DIRS
      ${DICE_HEADER_DIRS}      
      ${CMAKE_CURRENT_SOURCE_DIR}/src/opencv
  )
  add_definitions(-DDICE_ENABLE_OPENCV=1)
ENDIF()

SET(DICE_LIBRARIES
    ${DICE_LIBRARIES}
    teuchoscore
    teuchosnumerics
    teuchoscomm
    teuchosparameterlist
 )

# enable tpetra if chosen:
IF(DICE_ENABLE_MANYCORE)
  MESSAGE(STATUS "** MANYCORE enabled (uses Tpetra and Kokkos libraries) **")
  SET(DICE_LIBRARIES
    ${DICE_LIBRARIES}
    kokkoscore
    tpetra
  )
  ADD_DEFINITIONS(-DDICE_KOKKOS=1 -DDICE_TPETRA=1)
ELSE()
  MESSAGE(STATUS "MANYCORE disabled")
ENDIF()

IF(DICE_ENABLE_GLOBAL)
  MESSAGE(STATUS "*** Enabling Global DIC (requires the Tpetra (or Epetra), Seacas libraries, and Belos in Trilinos) ***")
  IF(DICE_USE_TPETRA)
    SET(DICE_LIBRARIES
      ${DICE_LIBRARIES}
      tpetra
      exodus
      belos
      belostpetra
    )
    ADD_DEFINITIONS(-DDICE_TPETRA=1)
  ELSE()
    SET(DICE_LIBRARIES
      ${DICE_LIBRARIES}
      epetra
      exodus
      belos
      ifpack
      belosepetra
    )
  ENDIF()
ELSE()
  MESSAGE(STATUS "Global DIC will not be enabled (to enable, set -D DICE_ENABLE_GLOBAL:BOOL=ON in the CMake script)")
  SET(DICE_LIBRARIES
    ${DICE_LIBRARIES}
    epetra
  )
ENDIF()

# WINDOWS CMake has a bug for find_package() for clapack
# f2clibs have to be added manually here
IF(WIN32)
  SET(DICE_LIBRARIES ${DICE_LIBRARIES} libf2c)
ENDIF()

# FIND BOOST
IF(WIN32)
  find_package(Boost 1.46.0) # find_package for boost works a little dif for Windows
ELSE()
  set(Boost_USE_STATIC_LIBS   ON)
  find_package(Boost 1.46.0 COMPONENTS program_options system filesystem timer chrono)
ENDIF()

if (Boost_FOUND)
  find_library(tiff_lib NAMES libtiff.a tiff PATHS ${TIFF_DIR}/lib)
  if(tiff_lib)
    MESSAGE(STATUS "Tiff file format enabled. Using tiff library from ${tiff_lib}")
    SET(DICE_LIBRARIES ${DICE_LIBRARIES} diceutils ${tiff_lib})
    IF(WIN32)
      FILE(COPY ${TIFF_DIR}/bin/tiff.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
    ENDIF()
  else()
    MESSAGE(FATAL_ERROR "Could not find required tiff library.")
  endif()
  find_library(jpeg_lib NAMES libjpeg.a jpeg PATHS ${JPEG_DIR}/lib)
  if(jpeg_lib)
    MESSAGE(STATUS "Jpeg file format enabled. Using jpeg library from ${jpeg_lib}")
    SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${jpeg_lib})
    ADD_DEFINITIONS(-DDICE_JPEG=1)
    SET(DICE_HAS_JPEG ON)
  else()
    MESSAGE(STATUS "Could not find jpeg library. Jpeg file format is NOT enabled")
  endif()
  find_library(png_lib NAMES libpng.a png PATHS ${PNG_DIR}/lib)
  if(png_lib)
    MESSAGE(STATUS "Png file format enabled. Using png library from ${png_lib}")
    SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${png_lib})
    IF(WIN32)
      FILE(COPY ${PNG_DIR}/bin/libpng16.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
      FILE(COPY ${ZLIB_DIR}/bin/zlib.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
    ENDIF()
    ADD_DEFINITIONS(-DDICE_PNG=1)
    SET(DICE_HAS_PNG ON)
  else()
    MESSAGE("Could not find png library. Png file format is NOT enabled")
  endif()
  MESSAGE(STATUS "Using Boost libraries from: ${Boost_LIBRARY_DIRS} and inlcudes from: ${Boost_INCLUDE_DIRS}")
  link_directories(${Boost_LIBRARY_DIRS})
else (Boost_FOUND)
  MESSAGE(FATAL_ERROR "BOOST is required, but not found")
endif (Boost_FOUND)

IF(DICE_DISABLE_BOOST_FILESYSTEM)
  ADD_DEFINITIONS(-DDICE_DISABLE_BOOST_FILESYSTEM=1)
  MESSAGE(STATUS "BOOST Filesystem has been disabled, output will be written to current execution directory")
ENDIF()

# if debug messages are turned on:
IF(DICE_DEBUG_MSG)
  MESSAGE(STATUS "Debugging messages are ON")
  ADD_DEFINITIONS(-DDICE_DEBUG_MSG=1)
ELSE(DICE_DEBUG_MSG)
  MESSAGE(STATUS "Debugging messages are OFF")
ENDIF(DICE_DEBUG_MSG)

# Windows: use Trilinos compiler flags
# Linux: don't use compiler flags from Trilinos, instead set them manually
# but pick up openmp if Trilinos was compiled with it:
if(WIN32)
  SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})
  SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS}) 
Else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3")
  STRING(FIND ${Trilinos_CXX_COMPILER_FLAGS} "openmp" OpenMPFound)
  IF( ${OpenMPFound} GREATER -1 )
    MESSAGE(STATUS "OpenMP was enabled in Trilinos so enabling it here. (Found flag at position ${OpenMPFound})")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fopenmp")
  ENDIF()
  STRING(FIND ${Trilinos_CXX_COMPILER_FLAGS} "c++11" CXX11Found)
  IF( ${CXX11Found} GREATER -1 )
    MESSAGE(STATUS "c++11 was enabled in Trilinos so enabling it here. (Found flag at position ${CXX11Found})")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11")
  ELSE()
    MESSAGE(FATAL_ERROR "c++11 must be enabled for DICe and Trilinos")
  ENDIF()
endif()
MESSAGE(STATUS "Trilinos CMAKE_CXX_FLAGS: ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE(STATUS "Trilinos CMAKE_C_FLAGS: ${Trilinos_C_COMPILER_FLAGS}")
MESSAGE(STATUS "DICe CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "DICe CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

# Get the git information to put in the header message (to identify the commit corresponding
# to the executable that was run
execute_process(
  COMMAND git describe --abbrev=6 --dirty --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SHA1
    OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE(STATUS "Git sha1: ${GIT_SHA1}")
ADD_DEFINITIONS(-DGITSHA1=\"${GIT_SHA1}\")

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)
