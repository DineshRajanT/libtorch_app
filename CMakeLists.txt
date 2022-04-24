cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0054 NEW)
project(example-app)
set(CMAKE_PREFIX_PATH "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
#set(OpenCV_DIR "${OpenCV_INCLUDE_DIRS}")
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(Torch_DIR "C:/Users/anaconda3/envs/dl/lib/site-packages/torch/share/cmake/Torch")
#include_directories("C:/Users/anaconda3/envs/dl/lib/site-packages/torch")
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${OpenCV_LIBS}")
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)	
endif (MSVC)
