cmake_minimum_required(VERSION 3.10)

# Project name
project(YOLO_RealTime)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})


# Add the executable
add_executable(tiny_yolo_image tiny_yolo_image.cpp)

# Add the executable
add_executable(tiny_yolo_image_test tiny_yolo_image_test.cpp)

target_link_libraries(tiny_yolo_image ${OpenCV_LIBS} pthread)

target_link_libraries(tiny_yolo_image_test ${OpenCV_LIBS} pthread)