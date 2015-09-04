add_subdirectory(component)

enable_testing()

#  Component Tests:
#  Tests classes in DICE
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

# copy the image files to the build dir
FILE ( GLOB img_files "${CMAKE_CURRENT_SOURCE_DIR}/component/images/*.*")
FOREACH ( img_file ${img_files} )
    FILE(COPY ${img_file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/component/images/ )
ENDFOREACH ( )
