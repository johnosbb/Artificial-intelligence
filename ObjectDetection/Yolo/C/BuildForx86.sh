#!/usr/bin/bash
./CleanMe.sh
cmake -DCMAKE_BUILD_TYPE=Debug  .
cmake --build .
