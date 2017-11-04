
SET(SOURCES
  DICe_api.cpp
  )

SET(HEADERS
  DICe_api.h
  )

include_directories (
    . 
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

# Now build executables from this library, tailored main()s, and Trilinos


add_library(diceapi ${SOURCES} ${HEADERS})
target_link_libraries(diceapi ${DICE_LIBRARIES} dice) 
target_compile_definitions(diceapi PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS diceapi
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
install(FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

set_target_properties(diceapi dice PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
