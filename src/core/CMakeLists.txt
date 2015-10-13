
SET(DICE_SOURCES
  DICe_Kokkos.cpp
  DICe_Image.cpp
  DICe_Subset.cpp
  DICe_Shape.cpp
  DICe_Simplex.cpp
  DICe_Schema.cpp
  DICe_ObjectiveZNSSD.cpp
  DICe_Parser.cpp
  DICe_PostProcessor.cpp
  DICe_ParameterUtilities.cpp
  )

SET(DICE_HEADERS
  DICe.h
  DICe_Kokkos.h
  DICe_Image.h
  DICe_Subset.h
  DICe_Shape.h
  DICe_Simplex.h
  DICe_Objective.h
  DICe_ObjectiveZNSSD.h
  DICe_Schema.h
  DICe_Multifield.h
  DICe_ParameterUtilities.h
  DICe_Parser.h
  DICe_PostProcessor.h
  DICe_Utilities.h
  )

include_directories (
    .
    ../../src/utils/src/
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
include_directories(${Boost_INCLUDE_DIRS} COMPONENTS gil program_options REQUIRED)
MESSAGE(STATUS "Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")

add_library(dicecore ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicecore ${Boost_LIBRARIES} ${DICE_LIBRARIES} diceutils dicefft)
target_compile_definitions(dicecore PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecore
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
