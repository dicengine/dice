
SET(DICE_SOURCES
  DICe.cpp
  DICe_Image.cpp
  DICe_Subset.cpp
  DICe_Shape.cpp
  DICe_Simplex.cpp
  )

SET(DICE_HEADERS
  DICe.h
  DICe_Image.h
  DICe_Subset.h
  DICe_Shape.h
  DICe_Simplex.h
  )

IF(DICE_ENABLE_MANYCORE)
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    DICe_SubsetKokkos.cpp
    DICe_ImageKokkos.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    DICe_Kokkos.h
    DICe_MultiFieldTpetra.h
    DICe_SubsetFunctors.h
  )
ELSE()
  SET(DICE_SOURCES
    ${DICE_SOURCES}
    DICe_SubsetSerial.cpp
    DICe_ImageSerial.cpp
    DICe_ImageUtils.cpp
  )
  SET(DICE_HEADERS
    ${DICE_HEADERS}
    DICe_ImageUtils.h
    DICe_MultiFieldEpetra.h
  )
ENDIF()

include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)
include_directories(${Boost_INCLUDE_DIRS} COMPONENTS gil program_options REQUIRED)
MESSAGE(STATUS "Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS}")

add_library(dicebase ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dicebase ${Boost_LIBRARIES})
target_compile_definitions(dicebase PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dicebase
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
