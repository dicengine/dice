SET(DICE_UTILS_SOURCES
  DICe_ImageIO.cpp
  DICe_Rawi.cpp
  ../../cine/DICe_Cine.cpp
)
IF(DICE_ENABLE_NETCDF)
SET(DICE_UTILS_SOURCES
  ${DICE_UTILS_SOURCES}
  ../../netcdf/DICe_NetCDF.cpp
)
ENDIF()

SET(DICE_UTILS_HEADERS
  DICe_ImageIO.h
  DICe_Rawi.h
  ../../cine/DICe_Cine.h
)

IF(DICE_ENABLE_NETCDF)
SET(DICE_UTILS_HEADERS
  ${DICE_UTILS_HEADERS}
  ../../netcdf/DICe_NetCDF.h
)
ENDIF()

include_directories(
    .
    ../../core
    ../../base
    ../../cine
    ../../netcdf
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
    ${NetCDF_DIR}
)

SET(DICE_UTILS_LIBRARIES teuchoscore)
if(DICE_ENABLE_NETCDF)
  SET(DICE_UTILS_LIBRARIES ${DICE_UTILS_LIBRARIES} netcdf)
ENDIF()

# Specify target & source files to compile it from
add_library(
    diceutils
    ${DICE_UTILS_SOURCES}
    ${DICE_UTILS_HEADERS}
)

target_link_libraries(diceutils ${DICE_UTILS_LIBRARIES})
IF(DICE_ENABLE_OPENCV)
  target_link_libraries(diceutils ${OpenCV_LIBRARIES})
ENDIF()
target_compile_definitions(diceutils PRIVATE -D DICE_LIB_EXPORTS_MODE)
MESSAGE(STATUS "diceutils library will be output to location: ${DICE_LIB_INSTALL_PREFIX}")
set_target_properties(diceutils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

install(FILES ${DICE_UTILS_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(TARGETS diceutils
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)


