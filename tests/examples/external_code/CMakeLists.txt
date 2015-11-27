cmake_minimum_required(VERSION 2.8)

MESSAGE("\nConfiguring the external code example...\n")

PROJECT(external_code)

# Define the location of the dice libraries:
#
# NOTE: If this example is used as a template, this directory should be redefined
# or define DICE_LIB_DIR in your do-cmake script with -D DICE_LIB_DIR:FILE_PATH="<location>"
#
# In the current example, DICE_LIB_INSTALL_LOCATION gets passed to this
# cmake configuration by a call to external_project_add() which is done by the main
# DICe project when building this example as a test

SET(DICE_LIB_DIR ${DICE_OUTPUT_PREFIX}/lib)
MESSAGE(STATUS "Using DICe libraries from: ${DICE_LIB_DIR}")
link_directories(${DICE_LIB_DIR})

add_executable(main main.cpp)
target_link_libraries(main dicebase dicecore dicefft diceutils dicecine)
