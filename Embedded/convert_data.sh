#!/bin/bash
# This script requires xxd, to install: sudo apt-get -qq install xxd
# The sed instructioins guarantee that the model resides in the program memory (Flash) and that the array is aligned to the 8-byte boundary
# Since every byte matters on a constrianed memory device and because the SRAM has a limited capacity,
# we keep the model in program memory. This ia also generally more memory efficient when the weights are constant.

(cd ./models; xxd -i preventive_forecast.tflite > model.h; sed -i 's/unsigned char/const unsigned char/g' model.h; sed -i 's/const/alignas(8) const/g' model.h;cat model.h)
