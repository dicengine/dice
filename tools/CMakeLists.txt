
include_directories (
    . 
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

SET(DICE_TEST_LIBRARIES
   dicebase
   dicecore
   dicefft
   dicecine
)

if(DICE_ENABLE_GLOBAL)
SET(DICE_TEST_LIBRARIES
  ${DICE_TEST_LIBRARIES}
  dicemesh
  triangle)
endif()

# DICE_LIBRARIES has the Trilinos libs (teuchos, tiff)
# DICE_TEST_LIBRARIES has dicecore (Image, Subset, Schema...)

add_executable(DICe_Diff DICe_Diff.cpp)
target_link_libraries(DICe_Diff ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

install(TARGETS DICe_Diff
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

add_executable(DICe_DiffAvg DICe_DiffAvg.cpp)
target_link_libraries(DICe_DiffAvg ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES} teuchosparameterlist)

install(TARGETS DICe_DiffAvg
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

#if(DICE_ENABLE_GLOBAL)
#add_executable(DICe_TextToMesh DICe_TextToMesh.cpp)
#target_link_libraries(DICe_TextToMesh ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES} teuchosparameterlist)
#
#install(TARGETS DICe_TextToMesh
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#)
#set_target_properties(DICe_TextToMesh
#  PROPERTIES
#  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
#  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
#  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
#)
#endif()

add_executable(DICe_CineToTiff           DICe_CineToTiff.cpp)
target_link_libraries(DICe_CineToTiff    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

install(TARGETS DICe_CineToTiff
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

set_target_properties(DICe_CineToTiff DICe_Diff DICe_DiffAvg
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
