cmake_minimum_required(VERSION 3.10)

# Project Name
project(YOLO_Darknet_Embedded_Example)

# Set C standard
set(CMAKE_C_STANDARD 11)

# Specify the path to the Darknet library and headers
# Replace these paths with the actual paths where you installed or built Darknet
# set(DARKNET_LIB_PATH "/usr/local/lib/libdarknet.so")
# set(DARKNET_INCLUDE_PATH "/usr/local/include/darknet.h")

# Add the executable
add_executable(yolo_example main.c)

# # Include the Darknet headers
# include_directories(${DARKNET_INCLUDE_PATH})

# Link the Darknet library
target_link_libraries(yolo_example darknet m pthread)
