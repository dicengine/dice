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

#  AVERAGE_DIFF_TESTS
#
#  Run the dice executable given the input files
#  Only select files are copied from source to bin dir
#  The files copied are specified in the specific test dir CMakeLists.txt
#  An extra executable may need to be run to compare the output to a known solution
#  The solutions are averaged in a particular direction and compared to a command solution

SET(AVERAGE_DIFF_TESTS
  dic_challenge_5_znssd
  dic_challenge_14_vsg
  dic_challenge_14_nlvc
  dic_challenge_14_keys4
  #dic_challenge_15
)
FOREACH ( test ${AVERAGE_DIFF_TESTS})
# create a test
    ADD_TEST ( NAME "RUN_${test}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../src/core/dice -i input.xml -v)
    add_subdirectory(${test})
			      
    # run the verification test for each output created
    FILE ( GLOB avg_diff_files "${CMAKE_CURRENT_SOURCE_DIR}/${test}/*.avg_diff")
    FOREACH ( avg_diff_file ${avg_diff_files})
        # get the name of the test from the filename
        GET_FILENAME_COMPONENT ( name ${avg_diff_file} NAME_WE)
        add_test(NAME "DIFF_${test}_${name}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}/
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../tools/DICe_DiffAvg ${avg_diff_file})
        set_tests_properties("DIFF_${test}_${name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_${test}_${name}" APPEND PROPERTY DEPENDS "RUN_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )

