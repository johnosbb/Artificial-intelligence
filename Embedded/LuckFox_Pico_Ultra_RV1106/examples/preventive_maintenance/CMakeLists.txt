cmake_minimum_required(VERSION 3.10)
project(pm_example C)

set(CMAKE_C_STANDARD 99)

# Set the RKNN API directory paths
set(RKNN_INCLUDE_DIR "/mnt/500GB/RKNN/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include")
set(RKNN_LIB_DIR "/mnt/500GB/RKNN/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc")

# Add the executable target
add_executable(pm_example main.c)

# Include the RKNN API headers
target_include_directories(pm_example PRIVATE ${RKNN_INCLUDE_DIR})

# Link the RKNN runtime libraries (use the available librknnmrt.so instead of librknnrt.so)
target_link_libraries(pm_example ${RKNN_LIB_DIR}/librknnmrt.so)
