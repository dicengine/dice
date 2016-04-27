SET(SOURCES
  triangle.c
  )

SET(HEADERS
  triangle.h
  )

include_directories (
    .
)

#add_definitions(-DTRILIBRARY -DANSI_DECLARATORS)

add_library(triangle ${SOURCES} ${HEADERS})
target_compile_definitions(triangle PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS triangle
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set_target_properties(triangle PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
)
