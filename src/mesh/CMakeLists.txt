
SET(DICE_SOURCES
  DICe_MeshEnums.cpp
  DICe_Mesh.cpp
  DICe_MeshIO.cpp
  )

SET(DICE_HEADERS
  DICe_MeshEnums.h
  DICe_Mesh.h
  DICe_MeshIO.h
  )

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

include_directories(${Boost_INCLUDE_DIRS})
add_library(dicemesh ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicemesh ${DICE_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(dicemesh ${Boost_LIBRARIES})
