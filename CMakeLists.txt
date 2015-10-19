cmake_minimum_required(VERSION 2.8)
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

SET(DICE_MPI_EXEC ${Trilinos_MPI_EXEC})

# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER opton

PROJECT(DICe)

include(ExternalProject)
MESSAGE(STATUS "Building DICe_utils from ${CMAKE_CURRENT_SOURCE_DIR}")
ExternalProject_Add(DICe_utils
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils_build
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils_tmp
  CMAKE_CACHE_ARGS -DDICE_TRILINOS_DIR:STRING=${DICE_TRILINOS_DIR} -DCMAKE_INSTALL_PREFIX:FILEPATH=${CMAKE_INSTALL_PREFIX} -DBOOST_ROOT:STRING=${BOOST_ROOT} -DTIFF_DIR:STRING=${TIFF_DIR} -DDICE_DEBUG_MSG:BOOL=${DICE_DEBUG_MSG} -DCLAPACK_DIR:FILEPATH=${CLAPACK_DIR} -DDICE_USE_DOUBLE:BOOL=${DICE_USE_DOUBLE}
  )

# base data type:
if(DICE_USE_DOUBLE)
  ADD_DEFINITIONS(-DDICE_USE_DOUBLE)
  MESSAGE(STATUS "Image intensity and scalar type will be: DOUBLE")
else()
  MESSAGE(STATUS "Image intensity and scalar type will be: FLOAT (default)")
endif()

# MPI check -- defaults to TRUE
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(DICE_MPI TRUE)
  ADD_DEFINITIONS(-DDICE_MPI)
  MESSAGE(STATUS "Using DICE_MPI_EXEC:${DICE_MPI_EXEC}")
ELSE()
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(DICE_MPI FALSE)
ENDIF()

SET(DICE_TRILINOS_HEADERS
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
  )

SET(DICE_TRILINOS_LIB_DIRS
    ${Trilinos_LIBRARY_DIRS}
    ${Trilinos_TPL_LIBRARY_DIRS}
  )

link_directories(${DICE_TRILINOS_LIB_DIRS} ${TIFF_DIR} ${CMAKE_INSTALL_PREFIX}/lib)

SET(DICE_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cine
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fft
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/src
  ${TIFF_DIR}
  ${TIFF_DIR}/../include
  )


SET(DICE_LIBRARIES
    ${DICE_LIBRARIES}
    teuchoscore
    teuchosnumerics
    teuchoscomm
    teuchosparameterlist
    kokkoscore
    tpetra
 )

# WINDOWS CMake has a bug for find_package() for clapack
# f2clibs have to be added manually here
IF(WIN32)
  SET(DICE_LIBRARIES ${DICE_LIBRARIES} libf2c)
ENDIF()

# FIND BOOST
IF(WIN32)
  find_package(Boost 1.46.0) # find_package for boost works a little dif for Windows
ELSE()
  find_package(Boost 1.46.0 COMPONENTS program_options system filesystem timer)
ENDIF()


if (Boost_FOUND)
  find_library(tiff_lib NAMES tiff PATHS ${TIFF_DIR})
  MESSAGE(STATUS "Using tiff library from: ${tiff_lib}")
  SET(DICE_LIBRARIES ${DICE_LIBRARIES} ${tiff_lib})
  link_directories(${Boost_LIBRARY_DIRS})
else (Boost_FOUND)
  MESSAGE(FATAL_ERROR "BOOST is required, but not found")
endif (Boost_FOUND)

IF(DICE_DISABLE_BOOST_FILESYSTEM)
  ADD_DEFINITIONS(-DDICE_DISABLE_BOOST_FILESYSTEM)
  MESSAGE(STATUS "BOOST Filesystem has been disabled, output will be written to current execution directory")
ENDIF()


# if debug messages are turned on:
IF(DICE_DEBUG_MSG)
  MESSAGE(STATUS "Debugging messages are ON")
  ADD_DEFINITIONS(-DDICE_DEBUG_MSG)
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
MESSAGE(STATUS "Trilnos CMAKE_C_FLAGS: ${Trilinos_C_COMPILER_FLAGS}")
MESSAGE(STATUS "DICe CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "DICe CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)

