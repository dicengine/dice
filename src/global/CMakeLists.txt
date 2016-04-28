SET(SOURCES
  DICe_GlobalUtils.cpp
)

SET(HEADERS
  DICe_GlobalUtils.h
)

include_directories (
  .
  ${DICE_HEADER_DIRS}
  ${DICE_TRILINOS_HEADERS}
)

add_library(diceglobal ${SOURCES} ${HEADERS})
target_compile_definitions(diceglobal PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS diceglobal
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set_target_properties(diceglobal PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
)

#
#  Computational mesh utilities
#
add_subdirectory(mesh)

#
#  Delaunay triangulation library
#
add_subdirectory(triangle)
