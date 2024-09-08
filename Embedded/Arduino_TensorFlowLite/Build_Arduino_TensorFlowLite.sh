
ARDUINO_LIB_NAME='Arduino_TensorFlowLite'
#Remove .git files

rm -r $ARDUINO_LIB_NAME/.*
#Remove docs

rm -r $ARDUINO_LIB_NAME/docs
#Remove examples

rm -r $ARDUINO_LIB_NAME/examples
#Remove scripts

rm -r $ARDUINO_LIB_NAME/scripts/
#Remove the test_over_serial folder

rm -r $ARDUINO_LIB_NAME/src/test_over_serial/
#Remove all files in the peripherals/ folder except for peripherals.h, utility.h and utility_arduino.cpp

mkdir tl_temp;cp  $ARDUINO_LIB_NAME/src/peripherals/utility_arduino.cpp tl_temp
cp  $ARDUINO_LIB_NAME/src/peripherals/peripherals.h tl_temp/
cp  $ARDUINO_LIB_NAME/src/peripherals/utility.h tl_temp/
rm -rf $ARDUINO_LIB_NAME/src/peripherals/*
cp tl_temp/* $ARDUINO_LIB_NAME/src/peripherals/
rm -rf tl_temp

ZIP_FILENAME="$ARDUINO_LIB_NAME"".zip"
rm $ZIP_FILENAME # remove any existing one
zip -r $ZIP_FILENAME $ARDUINO_LIB_NAME -q
