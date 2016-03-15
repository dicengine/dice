SET(DICE_SOURCES
  main.cpp
  mainwindow.cpp
  qimageroiselector.cpp
  qselectionarea.cpp
  simpleqtvtk.cpp
  resources.qrc
)

SET(DICE_HEADERS
  DICe_InputVars.h
  mainwindow.h
  qimageroiselector.h
  qselectionarea.h
  qdebugstream.h
  simpleqtvtk.h
)

include_directories (
  .
  ${DICE_HEADER_DIRS}
  ${DICE_TRILINOS_HEADERS}
  ${QT_DIR}/include
)

add_executable(dice_gui ${DICE_SOURCES} ${DICE_HEADERS})
target_link_libraries(dice_gui Qt5::Widgets dicecore dicemain)
target_compile_definitions(dice_gui PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${DICE_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dice_gui
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

set_target_properties(dice_gui PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${DICE_OUTPUT_PREFIX}/bin"
)
