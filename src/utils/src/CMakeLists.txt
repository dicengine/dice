SET(DICE_UTILS_SOURCES
  DICe_Tiff.cpp
  DICe_Rawi.cpp
)

SET(DICE_UTILS_HEADERS
  DICe_Tiff.h
  DICe_Rawi.h
)

include_directories(
    .
    ../../core
    ${DICE_HEADER_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${DICE_TRILINOS_HEADERS}
    ${TIFF_INCLUDE_DIR}
)

link_directories (
    ${TIFF_LIB_DIR}
)
SET(DICE_UTILS_LIBRARIES tiff)

# Specify target & source files to compile it from
add_library(
    diceutils
    ${DICE_UTILS_SOURCES}
    ${DICE_UTILS_HEADERS}
)

target_link_libraries(diceutils ${DICE_UTILS_LIBRARIES} ${Boost_LIBRARIES})
target_compile_definitions(diceutils PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_UTILS_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(TARGETS diceutils
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)


