# Instructions

- Build the binary using the BuildForSTM32.sh script.
- Copy the yolo_example binary to the STM32. I normally create a directory in /root called Yolo.
- You will then need to copy the various resource and configuration files to the same directory
  - coco.names
  - image.jpg
  - yolov3-tiny.cfg
  - yolov3-tiny.weights
 
To run the example use: ./yolo_example <cfg-file> <weights-file> <class-file> <image-file>

Images, Weights and Configuration files can be found [here](https://github.com/AlexeyAB/darknet/tree/master/data). 




The program will produce the following output:


```
# ./yolo_example yolov3-tiny.cfg  yolov3-tiny.weights coco.names image.jpg
Using input size: 416x416 as per yolov3-tiny.cfg
 Try to load cfg: yolov3-tiny.cfg, weights: yolov3-tiny.weights, clear = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
   1 max                2x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16 0.003 BF
   2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32 0.399 BF
   3 max                2x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32 0.001 BF
   4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64 0.399 BF
   5 max                2x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64 0.001 BF
   6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128 0.399 BF
   7 max                2x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128 0.000 BF
   8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256 0.399 BF
   9 max                2x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256 0.000 BF
  10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  11 max                2x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.000 BF
  12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
  13 conv    256       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 256 0.089 BF
  14 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  15 conv    255       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 255 0.044 BF
  16 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
  17 route  13                                     ->   13 x  13 x 256
  18 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  19 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  20 route  19 8                                   ->   26 x  26 x 384
  21 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  22 conv    255       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 255 0.088 BF
  23 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
Total BFLOPS 5.571
avg_outputs = 341534
 Try to load weights: yolov3-tiny.weights
Loading weights from yolov3-tiny.weights...
 seen 64, trained: 32013 K-images (500 Kilo-batches_64)
Done! Loaded 24 layers from weights-file
Network loaded with 24 layers.
Original image dimensions: 640x424
Detected object: Class sheep, Probability 0.83, Box with normalised locations [Center: (0.78, 0.55) Width: 0.35 Height: 0.51]
Box in original image size pixel locations: Box [Center: (501.82, 347.49) Width: 226.71 Height: 215.69]
Detected object: Class dog, Probability 0.89, Box with normalised locations [Center: (0.22, 0.73) Width: 0.19 Height: 0.20]
Box in original image size pixel locations: Box [Center: (141.57, 422.24) Width: 121.81 Height: 86.42]
Detected object: Class dog, Probability 0.81, Box with normalised locations [Center: (0.22, 0.72) Width: 0.10 Height: 0.14]
Box in original image size pixel locations: Box [Center: (140.49, 420.26) Width: 62.31 Height: 61.36]
Detected object: Class person, Probability 0.98, Box with normalised locations [Center: (0.36, 0.56) Width: 0.15 Height: 0.68]
Box in original image size pixel locations: Box [Center: (229.56, 352.94) Width: 98.66 Height: 286.97]

Timing statistics:
Network loading time: 2.98 seconds
Image loading time: 0.89 seconds
Class names loading time: 0.01 seconds
Prediction time: 226.04 seconds
Detection and conversion time: 0.00 seconds
Total time: 229.91 seconds
Detection complete.
```


![image](https://github.com/user-attachments/assets/8be0fc5e-a32f-4513-b07c-1714f0595cce)
_target image_

![image](https://github.com/user-attachments/assets/e0cb7a16-1f26-4348-ac3f-93c4b593bf05)
_image with overlays_


## Analysis of the ouput

```
 0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
```

After the first layer we are left with an output of 416 x 416 x  16. The input representing the image  416 x 416 x 3 (the 3 is the colour components R,G and B) goes through a convolutional layer with 16 filters. Each filter extracts a specific feature map from the image (e.g., edges, textures, patterns) producing of 16 feature maps. Each of these 16 channels represents a different feature map that was detected by one of the 16 convolutional filters. A convolutional filter (e.g., 3×3×3 for RGB input) slides over the image and computes a dot product at each position, producing a single channel output per filter, so with 16 filters, you get 16 channels in the output. We can also see in this output that this process took 0.150 BF, which means this convolutional layer requires 0.150 billion floating-point operations to process the input.

A kernel (or filter) is a small matrix used to extract specific features from an image, such as edges or textures. In our case we can calculate the billion floating-point operations using the following formula: $FLOPs = 2 \cdot (k^2) \cdot C_{in} \cdot C_{out} \cdot W \cdot H$, where k is kernel size (3 x 3) aand C is the number of channels, and W, H is the width and height. This fives us:
``` 2 x (9) x 3 x 16 x 416 x 416 = 149,520,384 = 149,520,384 FLOPS```
and as:

$1 \text{ BFLOP} = 10^9 \text{ FLOPs}$

We have a final calculation of:

$BFLOPS = \frac{149,520,384}{1,000,000,000} = 0.150 \text{ BFLOPS}$.




 

