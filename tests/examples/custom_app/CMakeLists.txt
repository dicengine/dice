cmake_minimum_required(VERSION 3.4)

MESSAGE("\nConfiguring the custom_app example...\n")

PROJECT(custom_app)
MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

#
# Define the install location for Trilinos
#

MESSAGE(STATUS "Using Trilinos installed in: ${DICE_TRILINOS_DIR}")
IF (NOT DEFINED DICE_TRILINOS_DIR)
  MESSAGE(FATAL_ERROR "\nError: cmake must define DICE_TRILINOS_DIR:
  (-D DICE_TRILINOS_DIR:FILEPATH=<trilinos_install_prefix>)!")
ENDIF()
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
SET(CMAKE_VERBOSE_MAKEFILE OFF)
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(DICE_MPI TRUE)
  ADD_DEFINITIONS(-DDICE_MPI=1)
ELSE()
  MESSAGE(STATUS "Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(DICE_MPI FALSE)
ENDIF()
SET(DICE_TRILINOS_HEADERS
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS})
SET(DICE_TRILINOS_LIB_DIRS
    ${Trilinos_LIBRARY_DIRS}
    ${Trilinos_TPL_LIBRARY_DIRS})
link_directories(${DICE_TRILINOS_LIB_DIRS})
include_directories(${DICE_TRILINOS_HEADERS})
SET(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})
SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS})
MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

# base data type:
if(DICE_USE_DOUBLE)
  ADD_DEFINITIONS(-DDICE_USE_DOUBLE=1)
  MESSAGE(STATUS "Image intensity and scalar type will be: DOUBLE")
else()
  MESSAGE(STATUS "Image intensity and scalar type will be: FLOAT (default)")
endif()

set(DICE_ENABLE_NETCDF OFF)
if(DEFINED NetCDF_DIR AND NOT "${NetCDF_DIR}" STREQUAL "")
  set(DICE_ENABLE_NETCDF ON)
  MESSAGE(STATUS "Using NetCDF from dir: ${NetCDF_DIR}")
  find_library(NetCDF_lib NAMES libnetcdf.a netcdf PATHS ${NetCDF_DIR})
  MESSAGE(STATUS "Using NetCDF lib: ${NetCDF_lib}")
  ADD_DEFINITIONS(-DDICE_ENABLE_NETCDF=1)
else()
  MESSAGE(STATUS "NetCDF will NOT be enabled")
endif()

#
# Define the location of the dice libraries, DICE_LIB_DIR:
#

# NOTE: If this example is used as a template, the DICE_LIB_DIR variable should be defined
# in your do-cmake script with -D DICE_LIB_DIR:STRING="<location>"
IF (NOT DEFINED DICE_LIB_DIR)
  MESSAGE(FATAL_ERROR "\nError: cmake must define DICE_LIB_DIR:
  (-D DICE_LIB_DIR:FILEPATH=<DICe_lib_prefix>)!")
ENDIF()
MESSAGE(STATUS "Using DICe libraries from: ${DICE_LIB_DIR}")
link_directories(${DICE_LIB_DIR})

#
# Define the location of the dice libraries, DICE_HEADER_DIR:
#

# NOTE: If this example is used as a template, the DICE_HEADER_DIR variable should be defined
# in your do-cmake script with -D DICE_HEADER_DIR:STRING="<location>"
IF (NOT DEFINED DICE_HEADER_DIR)
  MESSAGE(FATAL_ERROR "\nError: cmake must define DICE_HEADER_DIR:
  (-D DICE_HEADER_DIR:FILEPATH=<DICe_include_prefix>)!")
ENDIF()
MESSAGE(STATUS "Using DICe headers from: ${DICE_HEADER_DIR}")
include_directories(${DICE_HEADER_DIR})
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
if(WIN32)
link_directories(${CMAKE_INSTALL_PREFIX}/bin)
endif()
set(libprefix "lib")
set(libsuffix ".a")
if(WIN32)
  set(libprefix "")
  set(libsuffix ".lib")
endif()

add_executable(main main.cpp)
# add the dice libraries
target_link_libraries(main
  dicecore
  diceioutils
  "${libprefix}hypercine${libsuffix}"
)

# find OpenCV
message(STATUS "Looking for OpenCV in dir: ${OpenCV_DIR}")
find_package( OpenCV NO_DEFAULT_PATH PATHS ${OpenCV_DIR} )
MESSAGE(STATUS "Checking for OpenCV contrib libraries in ${OpenCV_DIR}/modules/xfeatures2d")
if(EXISTS "${OpenCV_DIR}/modules/xfeatures2d")
  MESSAGE(STATUS "OpenCV contrib modules found")
  add_definitions(-DHAVE_OPENCV_CONTRIB)
else()
  MESSAGE(STATUS "OpenCV contrib modules NOT found")
endif()

IF(OpenCV_FOUND)
  set(DICE_ENABLE_OPENCV ON)
ELSE()
  message(STATUS "OpenCV not found")
ENDIF()

IF(DICE_ENABLE_OPENCV)
  SET(DICE_HEADER_DIRS
      ${DICE_HEADER_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/opencv
  )
  add_definitions(-DDICE_ENABLE_OPENCV=1)
ENDIF()

IF(DICE_ENABLE_NETCDF)
  find_library(HDF5_lib NAMES libhdf5.a hdf5 PATHS ${HDF5_DIR} NO_DEFAULT_PATH)
  find_library(HDF5_lib NAMES libhdf5.a hdf5 PATHS ${HDF5_DIR})
  find_library(HDF5_hl_lib NAMES libhdf5_hl.a hdf5_hl PATHS ${HDF5_DIR} NO_DEFAULT_PATH)
  find_library(HDF5_hl_lib NAMES libhdf5_hl.a hdf5_hl PATHS ${HDF5_DIR})
  find_library(szip_lib NAMES libsz.a PATHS ${HDF5_DIR})
  MESSAGE(STATUS "Using HDF5 libs: ${HDF5_lib} ${HDF5_hl_lib}")
  MESSAGE(STATUS "Using szip libs: ${szip_lib}")
  target_link_libraries(main ${HDF5_lib} ${HDF5_hl_lib} ${NetCDF_lib})
  IF(szip_lib)
    target_link_libraries(main ${szip_lib})
  ENDIF()
ENDIF()

IF(DICE_ENABLE_GLOBAL)
  add_definitions(-DDICE_ENABLE_GLOBAL=1)
  target_link_libraries(main
  ${SEACAS_LIBRARIES}
  ${Ifpack_LIBRARIES}
  ${Belos_LIBRARIES})
  IF(${Trilinos_VERSION_MAJOR} LESS_EQUAL 13)
    target_link_libraries(main exodus)
  ENDIF()
ENDIF()

IF(DICE_ENABLE_OPENCV)
  target_link_libraries(main
    ${OpenCV_LIBRARIES})
ENDIF()
  
# add the other libraries needed
target_link_libraries(main
  ${TeuchosCore_LIBRARIES}
  ${TeuchosNumerics_LIBRARIES}
  ${TeuchosComm_LIBRARIES}
  ${TeuchosParameterList_LIBRARIES}
#  ${tiff_lib}
)
target_link_libraries(main ${epetra_LIBRARIES})
IF(DICE_ENABLE_GLOBAL)
  target_link_libraries(main
    ${BelosEpetra_LIBRARIES})
ENDIF()
