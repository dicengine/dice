cmake_minimum_required(VERSION 2.8)
MESSAGE("\nConfiguring DICe GUI...\n")

IF (NOT DEFINED DICE_TRILINOS_DIR)
  MESSAGE(FATAL_ERROR "\nDICe Error: DICE_TRILINOS_DIR must be defined (-D DICE_TRILINOS_DIR:FILEPATH=<trilinos-install-path>)")
ENDIF()
MESSAGE(STATUS "Using Trilinos installed in: ${DICE_TRILINOS_DIR}")

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

PROJECT(DICe_GUI)

link_directories(${Trilinos_LIBRARY_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})
SET(TRILINOS_LIBRARIES
  teuchoscore
  teuchosparameterlist
)

IF (NOT DEFINED DICE_EXEC_PATH)
  MESSAGE(FATAL_ERROR "\nDICe Error: DICE_EXEC_PATH must be defined (-D DICE_EXEC_PATH:FILEPATH=<dice_exec_path>)")
ENDIF()

MESSAGE(STATUS "Using DICe executable from ${DICE_EXEC_PATH}")
ADD_DEFINITIONS(-DDICE_EXEC_PATH=\"${DICE_EXEC_PATH}\")

# find the Qt libraries
MESSAGE(STATUS "Searching for Qt in dir: ${QT_DIR}")

find_package(Qt5Core PATHS ${QT_DIR})
IF(Qt5Core_FOUND)
  MESSAGE(STATUS "Found Qt5Core in ${QT_DIR}")
  link_directories(${QT_DIR}/lib)
ELSE()
  MESSAGE(FATAL_ERROR "\nERROR: could not find Qt5Core in the following directory: ${QT_DIR}")
ENDIF()

find_package(Qt5Widgets PATHS ${QT_DIR})
IF(Qt5Widgets_FOUND)
  MESSAGE(STATUS "Found Qt5Widgets in ${QT_DIR}")
ELSE()
  MESSAGE(FATAL_ERROR "\nERROR: could not find Qt5Widgets in the following directory: ${QT_DIR}")
ENDIF()

find_package(Qt5PrintSupport PATHS ${QT_DIR})
IF(Qt5PrintSupport_FOUND)
  MESSAGE(STATUS "Found Qt5PrintSupport in ${QT_DIR}")
ELSE()
  MESSAGE(FATAL_ERROR "\nERROR: could not find Qt5PrintSupport in the following directory: ${QT_DIR}")
ENDIF()

set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-arcs -ftest-coverage")

# set up a mapping so that the Release configuration for the Qt imported target is
# used in the COVERAGE CMake configuration.
set_target_properties(Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")
set_target_properties(Qt5::Widgets PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")
set_target_properties(Qt5::PrintSupport PROPERTIES MAP_IMPORTED_CONFIG_COVERAGE "RELEASE")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(VTK REQUIRED PATHS ${VTK_DIR})
IF(VTK_FOUND)
  MESSAGE(STATUS "Found VTK in ${VTK_DIR}")
  ELSE()
    MESSAGE(FATAL_ERROR "\nERROR: could not find VTK in the following directory: ${VTK_DIR}")
ENDIF()

# copy over the dll files so that they are in the path for windows
IF(WIN32)
  FILE(COPY ${QT_DIR}/bin/Qt5Core.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5Gui.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5Widgets.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5PrintSupport.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
ENDIF()

# Get the git information to put in the header message (to identify the commit corresponding
# to the executable that was run
execute_process(
  COMMAND git describe --abbrev=6 --dirty --always --tags
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_SHA1
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  MESSAGE(STATUS "Git sha1: ${GIT_SHA1}")
ADD_DEFINITIONS(-DGITSHA1=\"${GIT_SHA1}\")

add_subdirectory(DICe_2D)
