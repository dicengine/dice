MESSAGE(STATUS "Cleaning DICe CMake files")

set(cmake_generated ${CMAKE_BINARY_DIR}/dice/build/CMakeCache.txt
                    ${CMAKE_BINARY_DIR}/dice/build/cmake_install.cmake
                    ${CMAKE_BINARY_DIR}/dice/build/Makefile
                    ${CMAKE_BINARY_DIR}/dice/build/CMakeFiles
)

foreach(file ${cmake_generated})

  if (EXISTS ${file})
     file(REMOVE_RECURSE ${file})
  endif()

endforeach(file)