SET(DICE_UTILS_SOURCES
  ./DICe_ImageIO.cpp
  ../rawi/DICe_Rawi.cpp
)
IF(DICE_ENABLE_NETCDF)
SET(DICE_UTILS_SOURCES
  ${DICE_UTILS_SOURCES}
  ../netcdf/DICe_NetCDF.cpp
)
ENDIF()

SET(DICE_UTILS_HEADERS
  ./DICe_ImageIO.h
  ../rawi/DICe_Rawi.h
)

IF(DICE_ENABLE_NETCDF)
SET(DICE_UTILS_HEADERS
  ${DICE_UTILS_HEADERS}
  ../netcdf/DICe_NetCDF.h
)
ENDIF()

include_directories(
    .
    ../core
    ../base
    ../rawi
    ../netcdf
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
    ${NetCDF_DIR}
)

SET(DICE_UTILS_LIBRARIES teuchoscore teuchosparameterlist)
if(DICE_ENABLE_NETCDF)
  SET(DICE_UTILS_LIBRARIES ${DICE_UTILS_LIBRARIES} netcdf)
ENDIF()

# Specify target & source files to compile it from
add_library(
    diceioutils
    ${DICE_UTILS_SOURCES}
    ${DICE_UTILS_HEADERS}
)
add_dependencies(diceioutils hypercine)

target_link_libraries(diceioutils ${DICE_UTILS_LIBRARIES})
IF(DICE_ENABLE_OPENCV)
  target_link_libraries(diceioutils ${OpenCV_LIBRARIES})
ENDIF()
target_compile_definitions(diceioutils PRIVATE -D DICE_LIB_EXPORTS_MODE)
set_target_properties(diceioutils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

install(FILES ${DICE_UTILS_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(TARGETS diceioutils
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)


