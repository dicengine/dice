

#  STANDARD REGRESSION TESTS 
# 
#  Run the dice executable given the input files 
#  All files are copied from the source directory to the bin dir
#  The output files are diffed with gold copies
#
SET(REGRESSION_TESTS
  single_speckled_shift
  incremental
  conformal_subset_def
  dic_challenge_12
  dic_challenge_12_sorted
  simple_tracking
  reset_lost_subsets
)
IF(NOT UNIX)
  SET(REGRESSION_TESTS
    ${REGRESSION_TESTS}
    unicode_filename
  )
ENDIF()
IF(DICE_USE_DOUBLE)
  SET(REGRESSION_TESTS
    ${REGRESSION_TESTS}
    bitmap_star
    image_reg_init
  )
  IF(DICE_ENABLE_NETCDF AND NOT DICE_USE_INT_STORAGE)
    SET(REGRESSION_TESTS
      ${REGRESSION_TESTS}
      netcdf)
  ENDIF()
ELSE()
  SET(REGRESSION_TESTS
    ${REGRESSION_TESTS}
    # this test is flakey so turning it off
    # mechanism_optical_flow
    simple_obstruction
  )
ENDIF()


FOREACH ( test ${REGRESSION_TESTS})
    # Copy over all the test files (input and results)
    FILE(COPY ${test} DESTINATION . )
    # create a test
    ADD_TEST ( NAME "RUN_${test}"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    # remove the results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/results)
    # Separate test for each output file (could be more elegant way to do this...)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/${test}/gold/*.txt")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_${test}_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_Diff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name} -t 0.05 -f 1.0E-6 -v)
        set_tests_properties("DIFF_${test}_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_${test}_${trim_name}" APPEND PROPERTY DEPENDS "RUN_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )

#  Global method tests
#
#  Run the global formulation and compare the exodus output to a gold copy
#

IF(DICE_ENABLE_GLOBAL)
SET(GLOBAL_REGRESSION_TESTS
  global_14_unreg
  global_14_hs
  global_14_hs_mixed
  global_14_lm
  global_14_lm_mixed
  global_pixel_shift
  global_user_mesh
  14_initial_cond
  15_initial_cond
  stereo_cine
  stereo_list
  stereo_auto
  stereo_satellite
  stereo_dogbone
  stereo_dogbone_incremental
  )
IF(DICE_USE_DOUBLE)
SET(GLOBAL_REGRESSION_TESTS
 ${GLOBAL_REGRESSION_TESTS}
 stereo_d_spec
)
ENDIF()
FOREACH ( test ${GLOBAL_REGRESSION_TESTS})
    # Copy over all the test files (input and results)
    FILE(COPY ${test} DESTINATION . )
    # create a test
    ADD_TEST ( NAME "RUN_${test}"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    # remove the results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/results)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/projection_out.dat)
    # Separate test for each output file (could be more elegant way to do this...)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/${test}/gold/*.e")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_${test}_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_ExoDiff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name} ${CMAKE_CURRENT_BINARY_DIR}/${test}/exo_diff.txt)
        set_tests_properties("DIFF_${test}_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_${test}_${trim_name}" APPEND PROPERTY DEPENDS "RUN_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )

# add the strain executable tests
    # Copy over all the test files (input and results)
    FILE(COPY strain_exec DESTINATION . )
    # remove the old results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/DICe_multi_text*)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/DICe_single_text*)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/DICe_exo*)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/DICe_custom_exo*)
    # create a test    
    ADD_TEST ( NAME "RUN_strain_exec_DICe_multi_text"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/strain_exec
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice_strain DICe_solution_0.txt DICe_solution_1.txt DICe_solution_2.txt 25 StrainLocs.txt DICe_multi_text)
    ADD_TEST ( NAME "RUN_strain_exec_DICe_single_text"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/strain_exec
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice_strain DICe_solution_0.txt 25 DICe_single_text)
    ADD_TEST ( NAME "RUN_strain_exec_DICe_exo"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/strain_exec
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice_strain DICe_solution.e 25 DICe_exo)
    ADD_TEST ( NAME "RUN_strain_exec_DICe_custom_exo"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/strain_exec
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice_strain DICe_solution.e 25 StrainLocs.csv DICe_custom_exo)
    # Separate test for each output file (could be more elegant way to do this...)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/strain_exec/gold/*.e")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_strain_exec_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_ExoDiff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/${trim_name} ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/exo_diff.txt)
        set_tests_properties("DIFF_strain_exec_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
    ENDFOREACH ( )
    set_property(TEST "DIFF_strain_exec_DICe_multi_text.e" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_multi_text")
    set_property(TEST "DIFF_strain_exec_DICe_single_text.e" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_single_text")
    set_property(TEST "DIFF_strain_exec_DICe_exo.e" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_exo")
    set_property(TEST "DIFF_strain_exec_DICe_custom_exo.e" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_custom_exo")
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/strain_exec/gold/*.txt")
    FOREACH ( gold_file ${gold_files})
        # get the name of the file without the directory
        GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME)
        ADD_TEST ( NAME "DIFF_strain_exec_${trim_name}" 
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_Diff ${gold_file} ${CMAKE_CURRENT_BINARY_DIR}/strain_exec/${trim_name} -t 0.05 -v)
        set_tests_properties("DIFF_strain_exec_${trim_name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_strain_exec_${trim_name}" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_multi_text")
        set_property(TEST "DIFF_strain_exec_${trim_name}" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_single_text")
        set_property(TEST "DIFF_strain_exec_${trim_name}" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_exo")
        set_property(TEST "DIFF_strain_exec_${trim_name}" APPEND PROPERTY DEPENDS "RUN_strain_exec_DICe_custom_exo")
    ENDFOREACH ( )
    
ENDIF() # end global tests

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
  dic_challenge_15
)
FOREACH ( test ${AVERAGE_DIFF_TESTS})
# create a test
    ADD_TEST ( NAME "RUN_${test}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    add_subdirectory(${test})
    
    # run the verification test for each output created
    FILE ( GLOB avg_diff_files "${CMAKE_CURRENT_SOURCE_DIR}/${test}/*.avg_diff")
    FOREACH ( avg_diff_file ${avg_diff_files})
        # get the name of the test from the filename
        GET_FILENAME_COMPONENT ( name ${avg_diff_file} NAME_WE)
        add_test(NAME "DIFF_${test}_${name}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}/
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_DiffAvg ${avg_diff_file})
        set_tests_properties("DIFF_${test}_${name}" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_property(TEST "DIFF_${test}_${name}" APPEND PROPERTY DEPENDS "RUN_${test}")
    ENDFOREACH ( )
ENDFOREACH ( )

#
# MPI parallel tests
#

if(DICE_MPI)

  SET(MPI_TESTS
    parallel_subsets
    parallel_subsets_seed
    parallel_seed_first_step_only
  )
  IF(DICE_ENABLE_GLOBAL)
    SET(MPI_TESTS
      ${MPI_TESTS}
      stereo_d_spec
    )
  ENDIF()
  IF(DICE_USE_DOUBLE)
    SET(MPI_TESTS
      ${MPI_TESTS}
      parallel_image_def
    )
  ENDIF()
  FOREACH ( test ${MPI_TESTS})
    # Copy over all the test files (input and results)
    FILE(COPY ${test} DESTINATION . )
    # create a test
    ADD_TEST ( NAME "RUN_${test}_n2"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
      COMMAND ${DICE_MPI_EXEC} ${DICE_MPI_EXEC_OPTIONS} -n 2 ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    # Separate test for each output file (could be more elegant way to do this...)
    ADD_TEST ( NAME "RUN_${test}_n4"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test}
      COMMAND ${DICE_MPI_EXEC} ${DICE_MPI_EXEC_OPTIONS} -n 4 ${CMAKE_CURRENT_BINARY_DIR}/../../bin/dice -i input.xml -v)
    # Separate test for each output file (could be more elegant way to do this...)
    # remove the results folder (to clear previous runs)
    FILE(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/${test}/results)
    FILE ( GLOB gold_files "${CMAKE_CURRENT_BINARY_DIR}/${test}/gold/*.txt")
    FOREACH ( gold_file ${gold_files})
      # get the name of the file without the directory
      GET_FILENAME_COMPONENT ( trim_name ${gold_file} NAME_WE)
      ADD_TEST ( NAME "DIFF_${test}_${trim_name}_n2"
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_Diff ${gold_file} "${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name}" -t 1E-2 -f 1E-4 -v -p 2)
      ADD_TEST ( NAME "DIFF_${test}_${trim_name}_n4"
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/../../bin/DICe_Diff ${gold_file} "${CMAKE_CURRENT_BINARY_DIR}/${test}/results/${trim_name}" -t 1E-2 -f 1E-4 -v -p 4)
      set_tests_properties("DIFF_${test}_${trim_name}_n2" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
      set_tests_properties("DIFF_${test}_${trim_name}_n4" PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
      set_property(TEST "DIFF_${test}_${trim_name}_n2" APPEND PROPERTY DEPENDS "RUN_${test}_n2")
      set_property(TEST "DIFF_${test}_${trim_name}_n4" APPEND PROPERTY DEPENDS "RUN_${test}_n4")
    ENDFOREACH ( )
  ENDFOREACH ( )

endif(DICE_MPI)
