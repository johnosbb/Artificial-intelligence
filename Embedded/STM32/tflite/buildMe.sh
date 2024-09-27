#!/bin/bash
rm -rf CMakeCache.txt
cmake -DCMAKE_TOOLCHAIN_FILE=stm32_toolchain.cmake -DTFLITE_HOST_TOOLS_DIR=/usr/local/bin ../tensorflow/tensorflow/lite/

