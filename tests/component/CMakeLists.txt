include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
    
)

SET(DICE_COMP_LIBS
  dicecore
  diceapi
)

if(DICE_ENABLE_NETCDF)
  SET(DICE_COMP_LIBS
    ${DICE_COMP_LIBS}
    diceutils
    ${NetCDF_lib}
  )
ENDIF()

# glob all the cpp files
FILE ( GLOB test_cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
FOREACH ( test_file ${test_cpp_files})
    # get the name of the test from the cpp filename
    GET_FILENAME_COMPONENT ( name ${test_file} NAME_WE)
    add_executable(${name} ${test_file})
    target_link_libraries(${name} ${DICE_LIBRARIES} ${DICE_COMP_LIBS})
ENDFOREACH()

IF(DICE_ENABLE_NETCDF)
  add_subdirectory(netcdf)
ENDIF()

IF(DICE_ENABLE_OPENCV)
  add_subdirectory(opencv)
ENDIF()

IF(DICE_ENABLE_GLOBAL)
  add_subdirectory(global)
ENDIF()
