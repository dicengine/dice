
include_directories (
    . 
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
    ${Boost_INCLUDE_DIRS}
)

SET(DICE_TEST_LIBRARIES
   dicebase
   dicecore
   dicefft
   dicestereocalib
)

if(DICE_ENABLE_GLOBAL)
SET(DICE_TEST_LIBRARIES
  ${DICE_TEST_LIBRARIES}
  dicemesh
  triangle)
endif()

if(DICE_ENABLE_OPENCV)
SET(DICE_TEST_LIBRARIES
  ${DICE_TEST_LIBRARIES}
  dicefeature)
ENDIF()

IF(DICE_ENABLE_NETCDF)
  link_directories( ${NetCDF_DIR}/../lib )
ENDIF()

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

add_executable(DICe_CineStat           DICe_CineStat.cpp)
target_link_libraries(DICe_CineStat    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

install(TARGETS DICe_CineStat
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

add_executable(DICe_CrossInit           DICe_CrossInit.cpp)
target_link_libraries(DICe_CrossInit    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

install(TARGETS DICe_CrossInit
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

add_executable(DICe_Cal           DICe_Cal.cpp)
target_link_libraries(DICe_Cal    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

install(TARGETS DICe_Cal
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

set_target_properties(DICe_CineToTiff DICe_CineStat DICe_Diff DICe_DiffAvg DICe_CrossInit DICe_Cal
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

if(DICE_ENABLE_GLOBAL)
  add_executable(DICe_ExoDiff DICe_ExoDiff.cpp)
  target_link_libraries(DICe_ExoDiff ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES} dicemesh)

  install(TARGETS DICe_ExoDiff
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
  )
  set_target_properties(DICe_ExoDiff
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
  )
endif()
if(DICE_ENABLE_NETCDF)
  add_executable(DICe_NetCDFToTiff DICe_NetCDFToTiff.cpp)
  target_link_libraries(DICe_NetCDFToTiff ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES} dicebase)

  install(TARGETS DICe_NetCDFToTiff
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
  )
  set_target_properties(DICe_NetCDFToTiff
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
  )
endif()
