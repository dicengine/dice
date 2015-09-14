include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

add_executable(DICe_TestImage DICe_TestImage.cpp)
target_link_libraries(DICe_TestImage ${DICE_LIBRARIES} dicecore diceutils)

add_executable(DICe_TestSubset DICe_TestSubset.cpp)
target_link_libraries(DICe_TestSubset ${DICE_LIBRARIES} dicecore diceutils)

add_executable(DICe_TestSubsetGamma DICe_TestSubsetGamma.cpp)
target_link_libraries(DICe_TestSubsetGamma ${DICE_LIBRARIES} dicecore diceutils)
