
SET(DICE_SOURCES
  DICe_Image.cpp
  )

SET(DICE_HEADERS
  DICe.h
  DICe_KokkosTypes.h
  DICe_Image.h
  )

include_directories (
    .
    ../../src/utils/src/
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
include_directories(${Boost_INCLUDE_DIRS} COMPONENTS gil program_options REQUIRED)
MESSAGE(STATUS "Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")
IF(WIN32)
  SET(TIFF_DEF ${TIFF_DIR}/libtiff.def) # Windows likely didn't export symbols for libtiff so use .def file
ENDIF()

add_library(dicecore ${DICE_SOURCES} ${DICE_HEADERS} ${TIFF_DEF})
target_link_libraries(dicecore ${Boost_LIBRARIES})
target_compile_definitions(dicecore PRIVATE -D DICECORE_LIB_EXPORTS_MODE)
install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecore
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
