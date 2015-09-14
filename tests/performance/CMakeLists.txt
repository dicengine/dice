include_directories (
    .
    ../../src/utils/src/
    ${DICE_HEADER_DIRS}
    ${DICE_TRILINOS_HEADERS}
)

add_executable(DICe_PerformanceFunctors DICe_PerformanceFunctors.cpp)
target_link_libraries(DICe_PerformanceFunctors ${DICE_LIBRARIES} dicecore diceutils)
