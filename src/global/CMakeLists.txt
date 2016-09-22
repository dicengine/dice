SET(SOURCES
  DICe_Global.cpp
  DICe_GlobalUtils.cpp
  DICe_Preconditioner.cpp
  DICe_BCManager.cpp
)

SET(HEADERS
  DICe_Global.h
  DICe_GlobalUtils.h
  DICe_Preconditioner.h
  DICe_BCManager.h
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
#  Delaunay triangulation library
#
add_subdirectory(triangle)
