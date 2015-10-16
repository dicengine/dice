# Regression test command

#  STANDARD REGRESSION TESTS 
# 
#  Run the dice executable given the input files 
#  All files are copied from the source directory to the bin dir
#  The output files are diffed with gold copies
#
SET(REGRESSION_TESTS
#  single_speckled_shift
  conformal_subset_def
#  dic_challenge_12
#  obstructions
)

FOREACH ( test ${REGRESSION_TESTS})
    # Copy over all the test files (input and results)
    FILE(COPY ${test} DESTINATION . )
    # create a test
    ADD_TEST ( NAME "RUN_${test}"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../src/core/dice -i input.xml -v)
    # remove the results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/results)
    # Separate test for each output file (could be more elegant way to do this...)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/${test}/gold/*.txt")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_${test}_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../tools/DICe_Diff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name} -t 1E-4 -v)
        set_tests_properties("DIFF_${test}_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_${test}_${trim_name}" APPEND PROPERTY DEPENDS "RUN_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )
