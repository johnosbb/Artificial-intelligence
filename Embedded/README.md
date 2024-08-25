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
