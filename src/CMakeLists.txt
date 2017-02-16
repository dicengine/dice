#
#  Base (basic classes that do not have any dependencies outside of dice_base)
#
add_subdirectory(base)

#
#  Core (classes that build on the DICe base classes)
#
add_subdirectory(core)

#
#  Netcdf File Reader
#
IF(DICE_ENABLE_NETCDF)
  add_subdirectory(netcdf)
ENDIF()
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

IF(DICE_ENABLE_OPENCV)
  add_subdirectory(opencv)
ENDIF()
