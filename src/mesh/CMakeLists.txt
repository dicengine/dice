
SET(DICE_SOURCES
  DICe_MeshEnums.cpp
  DICe_Mesh.cpp
  )

SET(DICE_HEADERS
  DICe_MeshEnums.h
  DICe_Mesh.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

include_directories(${Boost_INCLUDE_DIRS})
add_library(dicemesh ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicemesh ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicebase)
target_compile_definitions(dicemesh PRIVATE -D DICE_LIB_EXPORTS_MODE)
install(TARGETS dicemesh
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set_target_properties(dicemesh PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

IF(DICE_ENABLE_GLOBAL)
  add_subdirectory(io)
ENDIF()
