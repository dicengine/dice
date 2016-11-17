
SET(DICE_SOURCES
  DICe_NetCDF.cpp
  )

SET(DICE_HEADERS
  DICe_NetCDF.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
    ${NetCDF_DIR}
)

add_library(dicenetcdf ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicenetcdf ${DICE_LIBRARIES})
target_compile_definitions(dicenetcdf PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicenetcdf
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

set_target_properties(dicenetcdf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
