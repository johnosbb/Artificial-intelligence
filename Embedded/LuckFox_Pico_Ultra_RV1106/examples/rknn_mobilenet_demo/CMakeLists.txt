cmake_minimum_required(VERSION 3.4.1)

project(rknn_mobilenet_demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# --- RKNN SDK Base Directory ---
# Set the common base directory for your RKNN toolkit installation.
# If you move the 'rknn-toolkit2' folder, you only need to change this one variable.
# Ensure this path is correct for your system.
set(RKNN_SDK_BASE_DIR "/mnt/500GB/RKNN")
message(STATUS "RKNN_SDK_BASE_DIR is set to: ${RKNN_SDK_BASE_DIR}")

# rknn api
# Construct RKNN_API_PATH using the base directory
set(RKNN_API_PATH "${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api")
message(STATUS "RKNN_API_PATH is set to: ${RKNN_API_PATH}")

set(RKNN_RT_LIB "${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so") # Use quotes for paths with variables

include_directories(${RKNN_API_PATH}/include)

# 3rdparty libraries (like stb)
# Construct THIRDPARTY_PATH using the base directory.
# This assumes the '3rdparty' folder is at '/mnt/500GB/RKNN/rknn-toolkit2/rknpu2/examples/3rdparty'
# If your '3rdparty' is in a different location relative to RKNN_SDK_BASE_DIR, adjust this path.
set(THIRDPARTY_PATH "${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/examples/3rdparty")
message(STATUS "THIRDPARTY_PATH is set to: ${THIRDPARTY_PATH}")
include_directories(${THIRDPARTY_PATH})


set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_mobilenet_demo
    src/main.cc
)

target_link_libraries(rknn_mobilenet_demo
    ${RKNN_RT_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_mobilenet_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_mobilenet_demo DESTINATION ./)

add_executable(rknn_mobilenet_nhwc_demo
    src/main_nhwc.cc
)

target_link_libraries(rknn_mobilenet_nhwc_demo
    ${RKNN_RT_LIB}
)

install(TARGETS rknn_mobilenet_nhwc_demo DESTINATION ./)

install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)