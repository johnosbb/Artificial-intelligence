#!/bin/bash
rm -rf CMakeCache.txt
cmake -DCMAKE_TOOLCHAIN_FILE=stm32_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DTFLITE_HOST_TOOLS_DIR=/usr/local/bin ../tensorflow/tensorflow/lite/examples/preventative_maintenance
