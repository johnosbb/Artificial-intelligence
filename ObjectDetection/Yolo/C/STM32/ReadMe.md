# Instructions

- Build the binary using the BuildForSTM32.sh script.
- Copy the yolo_example binary to the STM32. I normally create a directory in /root called Yolo.
- You will then need to copy the various resource files to the same directory
  - coco.names
  - image.jpg
  - yolov3-tiny.cfg
  - yolov3-tiny.weights
 
To run the example use: ./yolo_example <cfg-file> <weights-file> <class-file> <image-file>

Images, Weights and Configuration files can be found [here](https://github.com/AlexeyAB/darknet/tree/master/data). 


