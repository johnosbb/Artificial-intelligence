#!/bin/bash
rm -rf CMakeFiles cmake_install.cmake CMakeCache.txt Makefile pm_example
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=luckfox_toolchain.cmake .
make