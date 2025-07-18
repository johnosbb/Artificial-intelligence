cmake_minimum_required(VERSION 3.10)
project(snapshot)

set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_FIND_ROOT_PATH}/usr/include/opencv4")
link_directories("${CMAKE_FIND_ROOT_PATH}/usr/lib")

add_executable(still_capture main.cc)

target_link_libraries(still_capture  opencv_core opencv_imgproc opencv_imgcodecs)
