#
#  NetCDF component tests
#
SET(DICE_NETCDF_LIBS
  dicecore
  dicebase
  dicenetcdf
)
FILE ( GLOB test_netcdf_cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
FOREACH ( test_file ${test_netcdf_cpp_files})
  # get the name of the test from the cpp filename
  GET_FILENAME_COMPONENT ( name ${test_file} NAME_WE)
  add_executable(${name} ${test_file})
  target_link_libraries(${name} ${DICE_LIBRARIES} ${DICE_COMP_LIBS} ${DICE_NETCDF_LIBS})
ENDFOREACH()
