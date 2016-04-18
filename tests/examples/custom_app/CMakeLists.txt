cmake_minimum_required(VERSION 2.8)

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
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
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

#
# Bring in boost
#
# FIND BOOST
IF(WIN32)
  find_package(Boost 1.46.0) # find_package for boost works a little dif for Windows
ELSE()
  set(Boost_USE_STATIC_LIBS   ON)
  find_package(Boost 1.46.0 COMPONENTS program_options system filesystem timer)
ENDIF()
if (Boost_FOUND)
  MESSAGE(STATUS "Using Boost libraries from: ${Boost_LIBRARY_DIRS} and inlcudes from: ${Boost_INCLUDE_DIRS}")
  link_directories(${Boost_LIBRARY_DIRS})
else (Boost_FOUND)
  MESSAGE(FATAL_ERROR "BOOST is required, but not found")
endif (Boost_FOUND)

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

add_executable(main main.cpp)

#
# Define the location of lib tiff, jpeg, and png:
#

# NOTE: If this example is used as a template, the TIFF_DIR variable should be defined
# in your do-cmake script with -D TIFF_DIR:STRING="<location>" only if libtiff is not
# installed in a system path, otherwise it will get picked up by find_package
find_library(tiff_lib NAMES tiff PATHS ${TIFF_DIR}/lib)
if(NOT tiff_lib)
  MESSAGE(FATAL_ERROR "Tiff library is required, but not found")
else()
  MESSAGE(STATUS "Using tiff library from: ${tiff_lib}")
endif()

# NOTE: If this example is used as a template, the optional JPEG_DIR variable can be defined
# in your do-cmake script with -D JPEG_DIR:STRING="<location>" only if libjpeg is not
# installed in a system path, otherwise it will get picked up by find_package
find_library(jpeg_lib NAMES jpeg PATHS ${JPEG_DIR}/lib)
if(jpeg_lib)
  ADD_DEFINITIONS(-DDICE_JPEG=1)
  MESSAGE(STATUS "Using jpeg library from: ${jpeg_lib}")
  target_link_libraries(main ${jpeg_lib})
endif()

# NOTE: If this example is used as a template, the optional PNG_DIR variable can be defined
# in your do-cmake script with -D PNG_DIR:STRING="<location>" only if libjpeg is not
# installed in a system path, otherwise it will get picked up by find_package
find_library(png_lib NAMES png PATHS ${PNG_DIR}/lib)
if(png_lib)
  ADD_DEFINITIONS(-DDICE_PNG=1)
  MESSAGE(STATUS "Using png library from: ${png_lib}")
  target_link_libraries(main ${png_lib})
endif()

# add the dice libraries
target_link_libraries(main
  dicecore
  dicebase
  dicefft
  diceutils
  dicecine)
  
# add the other libraries needed
target_link_libraries(main
  teuchoscore
  teuchosnumerics
  teuchoscomm
  teuchosparameterlist
  ${tiff_lib}
  ${Boost_LIBRARIES})

IF(DICE_ENABLE_MANYCORE)
  target_link_libraries(main
    tpetra
    kokkoscore
  )
  ADD_DEFINITIONS(-DDICE_KOKKOS=1 -DDICE_TPETRA=1)
ELSE()
  target_link_libraries(main epetra)
ENDIF()
