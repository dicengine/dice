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

add_executable(DICe_TestCine DICe_TestCine.cpp)
target_link_libraries(DICe_TestCine ${DICE_LIBRARIES} dicecore diceutils dicecine)

add_executable(DICe_TestRawi DICe_TestRawi.cpp)
target_link_libraries(DICe_TestRawi ${DICE_LIBRARIES} dicecore diceutils)

add_executable(DICe_TestShape DICe_TestShape.cpp)
target_link_libraries(DICe_TestShape ${DICE_LIBRARIES} dicecore diceutils)

add_executable(DICe_TestFFT DICe_TestFFT.cpp)
target_link_libraries(DICe_TestFFT ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestPhaseCorrelation DICe_TestPhaseCorrelation.cpp)
target_link_libraries(DICe_TestPhaseCorrelation ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestObjective DICe_TestObjective.cpp)
target_link_libraries(DICe_TestObjective ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestSchema DICe_TestSchema.cpp)
target_link_libraries(DICe_TestSchema ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestParamUtils DICe_TestParamUtils.cpp)
target_link_libraries(DICe_TestParamUtils ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestTensor DICe_TestTensor.cpp)
target_link_libraries(DICe_TestTensor ${DICE_LIBRARIES} dicecore diceutils dicefft)

add_executable(DICe_TestCorrelateSquare DICe_TestCorrelateSquare.cpp)
target_link_libraries(DICe_TestCorrelateSquare ${DICE_LIBRARIES} dicecore diceutils dicefft)
