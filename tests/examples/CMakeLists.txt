#
#  Standard DICe use examples
#

SET(EXAMPLES
  mechanism
  obstruction
  full_field
)

FOREACH ( test ${EXAMPLES})
    # Copy over all the test files (input and results)
    FILE(COPY ${test} DESTINATION . )
    # create a test
    ADD_TEST ( NAME "RUN_EXAMPLE_${test}"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    # remove the results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/results)
    # Separate test for each output file (could be more elegant way to do this...)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/${test}/gold/*.txt")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_EXAMPLE_${test}_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_Diff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name} -t 0.05 -v)
        set_tests_properties("DIFF_EXAMPLE_${test}_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_EXAMPLE_${test}_${trim_name}" APPEND PROPERTY DEPENDS "RUN_EXAMPLE_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )

#
#  Examples using DICe in an external code by including DICe headers
#

# Copy over all the test files (input and results)
FILE(COPY external_code DESTINATION . )
# create a test
ADD_TEST ( NAME RUN_EXAMPLE_external_code
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external_code
   COMMAND ${CMAKE_CURRENT_BINARY_DIR}/external_code/build/main)
set_tests_properties(RUN_EXAMPLE_external_code PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")

# use an external projec to build the external_code example

include(ExternalProject)
MESSAGE(STATUS "Building external_code_example from ${CMAKE_CURRENT_SOURCE_DIR}")
ExternalProject_Add(external_code_example
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external_code
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_code
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external_code/build
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils_tmp
  CMAKE_CACHE_ARGS -DDICE_OUTPUT_PREFIX:STRING=${DICE_OUTPUT_PREFIX})
