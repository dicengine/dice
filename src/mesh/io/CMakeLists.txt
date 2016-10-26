include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

include_directories(${Boost_INCLUDE_DIRS})
add_library(dicemeshio DICe_MeshIO.cpp DICe_MeshIOUtils.cpp DICe_MeshIO.h DICe_MeshIOUtils.h)
target_link_libraries(dicemeshio ${DICE_LIBRARIES} ${Boost_LIBRARIES} dicemesh)
install(TARGETS dicemeshio
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
