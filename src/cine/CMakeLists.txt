
SET(DICE_SOURCES
  DICe_Cine.cpp
  )

SET(DICE_HEADERS
  DICe_Cine.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

add_library(dicecine ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicecine ${Boost_LIBRARIES} ${DICE_LIBRARIES})
target_compile_definitions(dicecine PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecine
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
