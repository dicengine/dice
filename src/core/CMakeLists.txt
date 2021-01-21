#
# Main standalone executable
#
add_executable(dice_main DICe_Main.cpp)
set_target_properties(dice_main PROPERTIES OUTPUT_NAME "dice")
target_link_libraries(dice_main ${DICE_LIBRARIES} dicecore diceioutils)
target_compile_definitions(dice_main PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS dice_main DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(dice_main PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)

IF(DICE_ENABLE_GLOBAL)
#
# Strain standalone executable
#
add_executable(dice_strain DICe_Strain.cpp)
set_target_properties(dice_strain PROPERTIES OUTPUT_NAME "dice_strain")
target_link_libraries(dice_strain ${DICE_LIBRARIES} dicecore diceioutils)
target_compile_definitions(dice_strain PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(TARGETS dice_strain DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(dice_strain PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
ENDIF()
