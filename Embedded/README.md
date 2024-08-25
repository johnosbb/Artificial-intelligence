# Building TensorFlow Lite for Arduino

## Check out the repository

```
ARDUINO_LIB_NAME='Arduino_TensorFlowLite'
git clone https://github.com/tensorflow/tflite-micro-arduino-examples.git $ARDUINO_LIB_NAME
```


## Checkout a Specific Hash

```
REF_TFLITE_MICRO_CK='0709653ed4938c49bd8072d75f07b93059375d04'
cd $ARDUINO_LIB_NAME; git checkout $REF_TFLITE_MICRO_CK
```

## Remove Unecessary Files

Remove .git files

```
rm -r $ARDUINO_LIB_NAME/.*
```

Remove docs

```
rm -r $ARDUINO_LIB_NAME/docs
```

Remove examples

```
rm -r $ARDUINO_LIB_NAME/examples
```

Remove scripts

```
rm -r $ARDUINO_LIB_NAME/scripts/
```

Remove the test_over_serial folder

```
rm -r $ARDUINO_LIB_NAME/src/test_over_serial/
```

Remove all files in the peripherals/ folder except for peripherals.h, utility.h and utility_arduino.cpp

```
mkdir tl_temp;cp  $ARDUINO_LIB_NAME/src/peripherals/utility_arduino.cpp tl_temp
cp  $ARDUINO_LIB_NAME/src/peripherals/peripherals.h tl_temp/
cp  $ARDUINO_LIB_NAME/src/peripherals/utility.h tl_temp/
rm -rf $ARDUINO_LIB_NAME/src/peripherals/*
cp tl_temp/* $ARDUINO_LIB_NAME/src/peripherals/
rm -rf tl_temp
```

Replace the peripherals.h file to work with all Arduino compatible microcontrollers

```
/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PERIPHERALS_H_
#define PERIPHERALS_H_

#ifdef ARDUINO
  #include <Arduino.h>
  #include <Wire.h>
#else  // ARDUINO
  #error "unsupported framework"
#endif  // ARDUINO

#include "utility.h"

#endif  // PERIPHERALS_H_
```


```
ZIP_FILENAME="$ARDUINO_LIB_NAME"".zip"
zip -r $ZIP_FILENAME $ARDUINO_LIB_NAME -q
```
