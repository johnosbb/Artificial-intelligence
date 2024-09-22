#!/bin/bash
# https://github.com/ccache/ccache/issues/1113
# https://android.googlesource.com/platform/external/tensorflow/+/6b511124eb0/tensorflow/lite/g3doc/guide/build_cmake.md
cmake -DCMAKE_TOOLCHAIN_FILE=stm32_toolchain.cmake -DTENSORFLOW_SOURCE_DIR=/mnt/500GB/tensorflow -DTFLITE_HOST_TOOLS_DIR=/usr/local/bin ..
cmake --build .
