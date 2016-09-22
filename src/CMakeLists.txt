#
#  Base (basic classes that do not have any dependencies outside of dice_base)
#
add_subdirectory(base)

#
#  Core (classes that build on the DICe base classes)
#
add_subdirectory(core)

#
#  Cine File Reader
#
add_subdirectory(cine)

#
# FFT utilities
#
add_subdirectory(fft)

#
# Dynamically linked library
#
add_subdirectory(api)

#
#  Computational mesh utilities
#
add_subdirectory(mesh)

IF(DICE_ENABLE_GLOBAL)
  add_subdirectory(global)
ENDIF()
