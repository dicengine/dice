

SET(DICE_SOURCES
  DICe_Simplex.cpp
  DICe_Schema.cpp
  DICe_ObjectiveZNSSD.cpp
  DICe_Parser.cpp
  DICe_XMLUtils.cpp
  DICe_PostProcessor.cpp
  DICe_ParameterUtilities.cpp
  DICe_Initializer.cpp
  )

SET(DICE_HEADERS
  DICe_Simplex.h
  DICe_Objective.h
  DICe_ObjectiveZNSSD.h
  DICe_Schema.h
  DICe_ParameterUtilities.h
  DICe_Parser.h
  DICe_XMLUtils.h
  DICe_PostProcessor.h
  DICe_Initializer.h
  DICe_Utilities.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
include_directories(${Boost_INCLUDE_DIRS} COMPONENTS gil program_options REQUIRED)
#MESSAGE(STATUS "Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")

add_library(dicecore ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicecore ${Boost_LIBRARIES} ${DICE_LIBRARIES} dicebase diceutils dicefft dicemesh)
IF(DICE_ENABLE_GLOBAL)
  target_link_libraries(dicecore diceglobal dicemeshio)
ENDIF()
target_compile_definitions(dicecore PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS} ../kdtree/nanoflann.hpp
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecore
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

#
# Main standalone executable
#
add_executable(dice_main DICe_Main.cpp)
set_target_properties(dice_main PROPERTIES OUTPUT_NAME "dice")
target_link_libraries(dice_main ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicebase dicecore dicecine)
target_compile_definitions(dice_main PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS dice_main DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(dicecore dice_main PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
