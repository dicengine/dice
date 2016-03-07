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

# copy over the dll files so that they are in the path for windows
IF(WIN32)
  FILE(COPY ${QT_DIR}/bin/Qt5Core.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5Gui.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5Widgets.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
  FILE(COPY ${QT_DIR}/bin/Qt5PrintSupport.dll DESTINATION ${DICE_OUTPUT_PREFIX}/bin)
ENDIF()

add_subdirectory(DICe_2D)
