
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

add_executable(DICe_CineToTiff           DICe_CineToTiff.cpp)
target_link_libraries(DICe_CineToTiff    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
install(TARGETS DICe_CineToTiff
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
