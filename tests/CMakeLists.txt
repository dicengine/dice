add_subdirectory(component)

enable_testing()

#  Component Tests:
#  Tests classes in DICe
#  Slightly more involved than a unit test, but very fine grained for one 
#  particular class

FILE ( GLOB test_cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/component/*.cpp")
FOREACH ( test_file ${test_cpp_files})
  # get the name of the test from the cpp filename
  GET_FILENAME_COMPONENT ( name ${test_file} NAME_WE)
  add_test(NAME ${name} 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/component
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/component/${name})
  set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
ENDFOREACH ( )

IF(DICE_ENABLE_GLOBAL)
  FILE ( GLOB test_global_cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/component/global/*.cpp")
  FOREACH ( test_file ${test_global_cpp_files})
    # get the name of the test from the cpp filename
    GET_FILENAME_COMPONENT ( name ${test_file} NAME_WE)
    add_test(NAME ${name} 
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/component/global/
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/component/global/${name})
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
  ENDFOREACH ( )
ENDIF()

# copy the image files to the build dir
FILE ( GLOB img_files "${CMAKE_CURRENT_SOURCE_DIR}/component/images/*.*")
FOREACH ( img_file ${img_files} )
    FILE(COPY ${img_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/component/images/ )
ENDFOREACH ( )

# copy the mesh files to the build dir
FILE ( GLOB mesh_files "${CMAKE_CURRENT_SOURCE_DIR}/component/global/meshes/*.*")
FOREACH ( mesh_file ${mesh_files} )
    FILE(COPY ${mesh_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/component/global/meshes/ )
ENDFOREACH ( )

#  Regression Tests:
#  Test DICe on full dataset analysis
#  These test use many of the components tested above
#
add_subdirectory(regression)

#  Performance Tests:
#  Tests the performance of DICe algorithms on the given architecture
#
add_subdirectory(performance)
# only one exectuble from the performance tests folder has an actual test run
ADD_TEST ( NAME "RUN_PerformanceFunctors"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}/performance
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/performance/DICe_PerformanceFunctors 1 1 -1)
set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")

# copy the image files to the build dir
FILE ( GLOB img_files "${CMAKE_CURRENT_SOURCE_DIR}/performance/images/*.*")
FOREACH ( img_file ${img_files} )
    FILE(COPY ${img_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/performance/images/ )
ENDFOREACH ( )

#  Examples:
#
#  These tests are used as tutorial examples
#
iF(NOT DICE_ENABLE_GLOBAL)
  add_subdirectory(examples)
endif()
