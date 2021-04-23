
include_directories (
    . 
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

SET(DICE_TEST_LIBRARIES
   dicecore
)

# DICE_LIBRARIES has the Trilinos libs (teuchos, tiff)
# DICE_TEST_LIBRARIES has dice (Image, Subset, Schema...)

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

#add_executable(DICe_InitCrossTool           DICe_InitCrossTool.cpp)
#target_link_libraries(DICe_InitCrossTool    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

#install(TARGETS DICe_InitCrossTool
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#)

#set_target_properties(DICe_InitCrossTool
#  PROPERTIES
#  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
#  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
#  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
#)


add_executable(DICe_Cal           DICe_Cal.cpp)
target_link_libraries(DICe_Cal    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

add_executable(DICe_Epiline           DICe_Epiline.cpp)
target_link_libraries(DICe_Epiline    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

set_target_properties(DICe_CineToTiff DICe_CineStat DICe_Diff DICe_DiffAvg DICe_CrossInit DICe_Cal DICe_Epiline
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

if(DICE_ENABLE_GLOBAL)
  add_executable(DICe_ExoDiff DICe_ExoDiff.cpp)
  target_link_libraries(DICe_ExoDiff ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  add_executable(DICe_ExoToFFT DICe_ExoToFFT.cpp)
  target_link_libraries(DICe_ExoToFFT ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  add_executable(DICe_2DExoImageDeformer DICe_2DExoImageDeformer.cpp)
  target_link_libraries(DICe_2DExoImageDeformer ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

  install(TARGETS DICe_ExoDiff DICe_2DExoImageDeformer DICe_ExoToFFT
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
  )
  set_target_properties(DICe_ExoDiff DICe_2DExoImageDeformer DICe_ExoToFFT
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
  )
endif()
if(DICE_ENABLE_NETCDF AND DICE_ENABLE_GLOBAL)
  add_executable(DICe_CalSatellite           DICe_CalSatellite.cpp)
  target_link_libraries(DICe_CalSatellite    ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  add_executable(DICe_NetCDFToTiff DICe_NetCDFToTiff.cpp)
  target_link_libraries(DICe_NetCDFToTiff ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  add_executable(DICe_ExoToNetCDF DICe_ExoToNetCDF.cpp)
  target_link_libraries(DICe_ExoToNetCDF ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})

  install(TARGETS DICe_NetCDFToTiff DICe_ExoToNetCDF DICe_CalSatellite
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
  )
  set_target_properties(DICe_NetCDFToTiff DICe_ExoToNetCDF DICe_CalSatellite
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
  )
endif()
if(DICE_ENABLE_OPENCV)
  add_executable(DICe_OpenCVServer DICe_OpenCVServer.cpp)
  target_link_libraries(DICe_OpenCVServer ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  add_executable(DICe_TrackingMovieMaker DICe_TrackingMovieMaker.cpp)
  target_link_libraries(DICe_TrackingMovieMaker ${DICE_LIBRARIES} ${DICE_TEST_LIBRARIES})
  install(TARGETS DICe_OpenCVServer DICe_TrackingMovieMaker
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
  )
  set_target_properties(DICe_OpenCVServer DICe_TrackingMovieMaker
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
  )
endif()
