include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

include_directories(${Boost_INCLUDE_DIRS})
add_library(dicemeshio DICe_MeshIO.cpp DICe_MeshIO.h)
target_link_libraries(dicemeshio ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicemesh)
target_link_libraries(dicemeshio ${Boost_LIBRARIES})
install(TARGETS dicemeshio
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
