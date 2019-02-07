


SET(DICE_SOURCES
  ./base/DICe.cpp
  ./base/DICe_Image.cpp
  ./base/DICe_Subset.cpp
  ./base/DICe_Shape.cpp
  ./base/DICe_FieldEnums.cpp
  ./base/DICe_LocalShapeFunction.cpp
  ./core/DICe_Camera.cpp
  ./core/DICe_CameraSystem.cpp
  ./core/DICe_Parser.cpp
  ./core/DICe_XMLUtils.cpp
  ./core/DICe_ParameterUtilities.cpp
  ./core/DICe_Simplex.cpp
  ./core/DICe_Schema.cpp
  ./core/DICe_Objective.cpp
  ./core/DICe_Triangulation.cpp
  ./core/DICe_PostProcessor.cpp
  ./core/DICe_Initializer.cpp
  ./core/DICe_Decomp.cpp
  ./fft/DICe_FFT.cpp
  ./fft/kiss_fft.c
  ./mesh/DICe_MeshEnums.cpp
  ./mesh/DICe_Mesh.cpp  
  )


SET(DICE_HEADERS
  ./base/DICe.h
  ./base/DICe_Image.h
  ./base/DICe_Subset.h
  ./base/DICe_Shape.h
  ./base/DICe_Matrix.h
  ./base/DICe_FieldEnums.h
  ./base/DICe_LocalShapeFunction.h
  ./core/DICe_Camera.h
  ./core/DICe_CameraSystem.h
  ./core/DICe_Parser.h
  ./core/DICe_XMLUtils.h
  ./core/DICe_PointCloud.h
  ./core/DICe_ParameterUtilities.h
  ./core/DICe_Simplex.h
  ./core/DICe_Objective.h
  ./core/DICe_Schema.h
  ./core/DICe_Triangulation.h
  ./core/DICe_PostProcessor.h
  ./core/DICe_Initializer.h
  ./core/DICe_Decomp.h
  ./kdtree/nanoflann.hpp
  ./fft/DICe_FFT.h
  ./fft/kiss_fft.h
  ./fft/_kiss_fft_guts.h
  ./mesh/DICe_MeshEnums.h
  ./mesh/DICe_Mesh.h
  )

IF(DICE_ENABLE_MANYCORE)
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    ./base/DICe_SubsetKokkos.cpp
    ./base/DICe_ImageKokkos.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    ./base/DICe_Kokkos.h
    ./base/DICe_MultiFieldTpetra.h
    ./base/DICe_SubsetFunctors.h
    ./base/DICe_ImageFunctors.h
  )
ELSE()
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    ./base/DICe_SubsetSerial.cpp
    ./base/DICe_ImageSerial.cpp
    ./base/DICe_ImageUtils.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    ./base/DICe_ImageUtils.h
    ./base/DICe_MultiFieldEpetra.h
  )
ENDIF()

IF(DICE_ENABLE_GLOBAL)
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    ./global/DICe_Global.cpp
    ./global/DICe_GlobalUtils.cpp
    ./global/DICe_Preconditioner.cpp
    ./global/DICe_BCManager.cpp
    ./global/triangle/triangle.c
    ./global/triangle/DICe_TriangleUtils.cpp
    ./mesh/io/DICe_MeshIO.cpp
    ./mesh/io/DICe_MeshIOUtils.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    ./global/DICe_Global.h
    ./global/DICe_GlobalUtils.h
    ./global/DICe_Preconditioner.h
    ./global/DICe_BCManager.h
    ./global/triangle/triangle.h
    ./global/triangle/DICe_TriangleUtils.h
    ./mesh/io/DICe_MeshIO.h
    ./mesh/io/DICe_MeshIOUtils.h
  )
  link_directories(${NetCDF_DIR}/../lib)
ENDIF()

IF(DICE_ENABLE_OPENCV)
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    ./opencv/DICe_Feature.cpp
    ./opencv/DICe_StereoCalib.cpp
    ./opencv/DICe_Calibration.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    ./opencv/DICe_Feature.h
    ./opencv/DICe_StereoCalib.h
    ./opencv/DICe_Calibration.h
  )
ENDIF()

include_directories (
    ./base
    ./core
    ./global
    ./global/triangle
    ./fft
    ./cine
    ./mesh
    ./mesh/io
    ./netcdf
    ./opencv
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
add_library(dicecore ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicecore ${DICE_LIBRARIES} ${OpenCV_LIBRARIES} diceutils)

target_compile_definitions(dicecore PRIVATE -D DICE_LIB_EXPORTS_MODE)
add_dependencies(dicecore DICe_utils)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicecore
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
set_target_properties(dicecore PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

#
#  Core (classes that build on the DICe base classes)
#
add_subdirectory(core)

#
# Dynamically linked library
#
add_subdirectory(api)


