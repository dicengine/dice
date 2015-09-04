include_directories (
    .
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

add_executable(DICe_TestImage DICe_TestImage.cpp)
target_link_libraries(DICe_TestImage ${DICE_LIBRARIES} dicecore diceutils)
