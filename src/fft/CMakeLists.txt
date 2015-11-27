
SET(FFT_SOURCES
  DICe_FFT.cpp
  kiss_fft.c
)

SET(FFT_HEADERS
  DICe_FFT.h
  kiss_fft.h
  _kiss_fft_guts.h
)


include_directories (
    ../base
    ../core
    .
    ${DICE_TRILINOS_HEADERS}
)

link_directories (
 ${DICE_TRILINOS_LIB_DIRS}
)

add_library(dicefft ${FFT_SOURCES} ${FFT_HEADERS})
target_link_libraries(dicefft ${DICE_LIBRARIES} dicebase)
target_compile_definitions(dicefft PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS	dicefft
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
install(FILES ${FFT_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

set_target_properties(dicefft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
