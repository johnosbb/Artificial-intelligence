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

Some core APIs do not have an implementation of RingBuffer, so it is best to add one. Create a RingBuffeTf.h in $ARDUINO_LIB_NAME/src/tensorflow/lite/micro/

```
/*
  Copyright (c) 2014 Arduino.  All right reserved.

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifdef __cplusplus

#ifndef _RING_BUFFER_TF_
#define _RING_BUFFER_TF_

#include <stdint.h>
#include <string.h>

namespace arduino {

// Define constants and variables for buffering incoming serial data.  We're
// using a ring buffer (I think), in which head is the index of the location
// to which to write the next incoming character and tail is the index of the
// location from which to read.
#define SERIAL_BUFFER_SIZE 64

template <int N>
class RingBufferNN
{
  public:
    uint8_t _aucBuffer[N] ;
    volatile int _iHead ;
    volatile int _iTail ;
    volatile int _numElems;

  public:
    RingBufferNN( void ) ;
    void store_char( uint8_t c ) ;
    void clear();
    int read_char();
    int available();
    int availableForStore();
    int peek();
    bool isFull();

  private:
    int nextIndex(int index);
    inline bool isEmpty() const { return (_numElems == 0); }
};

typedef RingBufferNN<SERIAL_BUFFER_SIZE> RingBuffer;


template <int N>
RingBufferNN<N>::RingBufferNN( void )
{
    memset( _aucBuffer, 0, N ) ;
    clear();
}

template <int N>
void RingBufferNN<N>::store_char( uint8_t c )
{
  // if we should be storing the received character into the location
  // just before the tail (meaning that the head would advance to the
  // current location of the tail), we're about to overflow the buffer
  // and so we don't write the character or advance the head.
  if (!isFull())
  {
    _aucBuffer[_iHead] = c ;
    _iHead = nextIndex(_iHead);
    _numElems++;
  }
}

template <int N>
void RingBufferNN<N>::clear()
{
  _iHead = 0;
  _iTail = 0;
  _numElems = 0;
}

template <int N>
int RingBufferNN<N>::read_char()
{
  if (isEmpty())
    return -1;

  uint8_t value = _aucBuffer[_iTail];
  _iTail = nextIndex(_iTail);
  _numElems--;

  return value;
}

template <int N>
int RingBufferNN<N>::available()
{
  return _numElems;
}

template <int N>
int RingBufferNN<N>::availableForStore()
{
  return (N - _numElems);
}

template <int N>
int RingBufferNN<N>::peek()
{
  if (isEmpty())
    return -1;

  return _aucBuffer[_iTail];
}

template <int N>
int RingBufferNN<N>::nextIndex(int index)
{
  return (uint32_t)(index + 1) % N;
}

template <int N>
bool RingBufferNN<N>::isFull()
{
  return (_numElems == N);
}

}

#endif /* _RING_BUFFER_TF_ */
#endif /* __cplusplus */
```

Then add a reference to the RingBufferTf.h file in $ARDUINO_LIB_NAME/src/tensorflow/lite/micro/system_setup.cpp

```
+ #include "tensorflow/lite/micro/RingBufferTf.h"
#include "tensorflow/lite/micro/system_setup.h"
#include <limits>

#include "tensorflow/lite/micro/debug_log.h"
```

And change the ring buffer reference in $ARDUINO_LIB_NAME/src/tensorflow/lite/micro/system_setup.cpp from RingBufferN to RingBufferNN

```
    class _RingBuffer : public RingBufferNN<kSerialMaxInputLength + 1>
    {
```

Create a Zip File

```
ZIP_FILENAME="$ARDUINO_LIB_NAME"".zip"
zip -r $ZIP_FILENAME $ARDUINO_LIB_NAME -q
```