SET(GUI_SOURCES
  main.cpp
  mainwindow.cpp
  simpleqtvtk.cpp
  resources.qrc
  ../../core/DICe_XMLUtils.cpp
)

SET(GUI_HEADERS
  DICe_InputVars.h
  mainwindow.h
  qdebugstream.h
  simpleqtvtk.h
)

include(${VTK_USE_FILE})

include_directories (
  .
  ${QT_DIR}/include
  ${VTK_DIR}/include/vtk-7.0
  ../../core
)

#  ${DICE_HEADER_DIRS}
#  ${DICE_TRILINOS_HEADERS}

add_executable(dice_gui ${GUI_SOURCES} ${GUI_HEADERS})
#target_link_libraries(dice_gui Qt5::Widgets dicecore dicemain ${VTK_LIBRARIES})
target_link_libraries(dice_gui Qt5::Widgets ${VTK_LIBRARIES})
#target_compile_definitions(dice_gui PRIVATE -D DICE_LIB_EXPORTS_MODE)

install(FILES ${GUI_HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
install(TARGETS dice_gui
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

set_target_properties(dice_gui PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin"
)
