
SET(DICE_SOURCES
  DICe_Kokkos.cpp
  DICe_Image.cpp
  DICe_Subset.cpp
  )

SET(DICE_HEADERS
  DICe.h
  DICe_Kokkos.h
  DICe_Image.h
  DICe_Subset.h
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
target_link_libraries(dicecore ${Boost_LIBRARIES} ${DICE_LIBRARIES} diceutils)
target_compile_definitions(dicecore PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecore
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
