SET(DICE_SOURCES
  mainwindow.cpp
  qimageroiselector.cpp
  qselectionarea.cpp
  resources.qrc
)

SET(DICE_HEADERS
  DICe_InputVars.h
  mainwindow.h
  qimageroiselector.h
  qselectionarea.h
)

include_directories (
  .
  ${DICE_HEADER_DIRS}
  ${DICE_TRILINOS_HEADERS}
  ${QT_DIR}/include
)

add_library(mainwindow ${DICE_SOURCES})
target_link_libraries (mainwindow Qt5::Widgets dicecore)

add_executable(dice_gui main.cpp resources.qrc)
target_link_libraries(dice_gui mainwindow)


install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS mainwindow
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
install(TARGETS dice_gui
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

set_target_properties(mainwindow dice_gui PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
