
SET(DICE_SOURCES
  DICe_Feature.cpp
  )

SET(DICE_HEADERS
  DICe_Feature.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

add_library(dicefeature ${DICE_SOURCES} ${DICE_HEADERS})

target_link_libraries(dicefeature ${DICE_LIBRARIES} ${OpenCV_LIBRARIES} dicebase)

target_compile_definitions(dicefeature PRIVATE -D DICE_LIB_EXPORTS_MODE)

add_library(dicestereocalib DICe_StereoCalib.cpp)

target_link_libraries(dicestereocalib ${DICE_LIBRARIES} ${OpenCV_LIBRARIES} dicebase)

target_compile_definitions(dicestereocalib PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(TARGETS dicefeature dicestereocalib
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

set_target_properties(dicefeature dicestereocalib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
