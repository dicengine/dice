SET(DICE_SOURCES_UTIL
  DICe_Parser.cpp
  DICe_XMLUtils.cpp
  DICe_ParameterUtilities.cpp
)
SET(DICE_HEADERS_UTIL
  DICe_Parser.h
  DICe_XMLUtils.h
  DICe_ParameterUtilities.h
)

SET(DICE_SOURCES
  DICe_Simplex.cpp
  DICe_Schema.cpp
  DICe_ObjectiveZNSSD.cpp
  DICe_Triangulation.cpp
  DICe_PostProcessor.cpp
  DICe_Initializer.cpp
  DICe_Decomp.cpp
  )

SET(DICE_HEADERS
  DICe_Simplex.h
  DICe_Objective.h
  DICe_ObjectiveZNSSD.h
  DICe_Schema.h
  DICe_Triangulation.h
  DICe_PostProcessor.h
  DICe_Initializer.h
  DICe_Utilities.h
  DICe_Decomp.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
include_directories(${Boost_INCLUDE_DIRS} COMPONENTS gil program_options REQUIRED)
#MESSAGE(STATUS "Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")

IF(DICE_ENABLE_NETCDF)
  link_directories( ${NetCDF_DIR}/../lib )
ENDIF()

add_library(diceparser ${DICE_SOURCES_UTIL} ${DICE_HEADERS_UTIL})
#target_link_libraries(diceparser ${Boost_LIBRARIES} teuchoscore)
target_compile_definitions(diceparser PRIVATE -D DICE_LIB_EXPORTS_MODE)
install(TARGETS diceparser
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set_target_properties(diceparser PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

add_library(dicecore ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicecore ${Boost_LIBRARIES} ${DICE_LIBRARIES} dicebase diceutils dicefft dicemesh diceparser)
IF(DICE_ENABLE_OPENCV)
  target_link_libraries(dicecore dicefeature)
ENDIF()
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
target_link_libraries(dice_main ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicebase dicecore diceutils diceparser)
IF(DICE_ENABLE_OPENCV)
  target_link_libraries(dice_main dicefeature)
ENDIF()
IF(DICE_ENABLE_GLOBAL)
  target_link_libraries(dice_main diceglobal)
ENDIF()
target_compile_definitions(dice_main PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS dice_main DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(dicecore dice_main PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)


IF(DICE_ENABLE_GLOBAL)
#
# Strain standalone executable
#
add_executable(dice_strain DICe_Strain.cpp)
set_target_properties(dice_strain PROPERTIES OUTPUT_NAME "dice_strain")
target_link_libraries(dice_strain ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicebase dicecore diceglobal dicefeature diceparser)
IF(DICE_ENABLE_OPENCV)
  target_link_libraries(dice_strain dicefeature)
ENDIF()
target_compile_definitions(dice_strain PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS dice_strain DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(dice_strain PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
ENDIF()
