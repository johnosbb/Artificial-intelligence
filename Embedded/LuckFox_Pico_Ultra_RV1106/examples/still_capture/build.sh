#!/bin/bash
rm -rf CMakeFiles cmake_install.cmake CMakeCache.txt Makefile still_capture
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=luckfox_toolchain.cmake .
make