#!/bin/bash
rm -rf CMakeFiles cmake_install.cmake CMakeCache.txt Makefile hello_world
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=luckfox_toolchain.cmake .
make