cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# --- RKNN SDK Base Directory ---
# Set the common base directory for your RKNN toolkit installation.
# If you move the 'rknn-toolkit2' folder, you only need to change this one variable.
set(RKNN_SDK_BASE_DIR "/mnt/500GB/RKNN")
message(STATUS "RKNN_SDK_BASE_DIR is set to: ${RKNN_SDK_BASE_DIR}")


# rknn api
# Construct RKNN_API_PATH using the base directory
set(RKNN_API_PATH "${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api")
message(STATUS "RKNN_API_PATH is set to: ${RKNN_API_PATH}")

set(RKNN_RT_LIB "${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so") # Use quotes for paths with variables

include_directories(${RKNN_API_PATH}/include)

# 3rdparty libraries (like stb)
# Construct THIRDPARTY_PATH using the base directory
set(THIRDPARTY_PATH "${RKNN_SDK_BASE_DIR}/rknn-toolkit2/rknpu2/examples/3rdparty")
message(STATUS "THIRDPARTY_PATH is set to: ${THIRDPARTY_PATH}")
include_directories(${THIRDPARTY_PATH})

# rknn_yolov5_demo local includes
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_demo
             src/main.cc
             src/postprocess.cc
)

target_link_libraries(rknn_yolov5_demo
${RKNN_RT_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5_demo DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)