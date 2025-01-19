# Object Detection on LuckFox

## Using Software Only

Using input size: 416x416 as per yolov3-tiny.cfg
Try to load cfg: yolov3-tiny.cfg, weights: yolov3-tiny.weights, clear = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
layer filters size/strd(dil) input output
0 conv 16 3 x 3/ 1 416 x 416 x 3 -> 416 x 416 x 16 0.150 BF
1 max 2x 2/ 2 416 x 416 x 16 -> 208 x 208 x 16 0.003 BF
2 conv 32 3 x 3/ 1 208 x 208 x 16 -> 208 x 208 x 32 0.399 BF
3 max 2x 2/ 2 208 x 208 x 32 -> 104 x 104 x 32 0.001 BF
4 conv 64 3 x 3/ 1 104 x 104 x 32 -> 104 x 104 x 64 0.399 BF
5 max 2x 2/ 2 104 x 104 x 64 -> 52 x 52 x 64 0.001 BF
6 conv 128 3 x 3/ 1 52 x 52 x 64 -> 52 x 52 x 128 0.399 BF
7 max 2x 2/ 2 52 x 52 x 128 -> 26 x 26 x 128 0.000 BF
8 conv 256 3 x 3/ 1 26 x 26 x 128 -> 26 x 26 x 256 0.399 BF
9 max 2x 2/ 2 26 x 26 x 256 -> 13 x 13 x 256 0.000 BF
10 conv 512 3 x 3/ 1 13 x 13 x 256 -> 13 x 13 x 512 0.399 BF
11 max 2x 2/ 1 13 x 13 x 512 -> 13 x 13 x 512 0.000 BF
12 conv 1024 3 x 3/ 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BF
13 conv 256 1 x 1/ 1 13 x 13 x1024 -> 13 x 13 x 256 0.089 BF
14 conv 512 3 x 3/ 1 13 x 13 x 256 -> 13 x 13 x 512 0.399 BF
15 conv 255 1 x 1/ 1 13 x 13 x 512 -> 13 x 13 x 255 0.044 BF
16 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
17 route 13 -> 13 x 13 x 256
18 conv 128 1 x 1/ 1 13 x 13 x 256 -> 13 x 13 x 128 0.011 BF
19 upsample 2x 13 x 13 x 128 -> 26 x 26 x 128
20 route 19 8 -> 26 x 26 x 384
21 conv 256 3 x 3/ 1 26 x 26 x 384 -> 26 x 26 x 256 1.196 BF
22 conv 255 1 x 1/ 1 26 x 26 x 256 -> 26 x 26 x 255 0.088 BF
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
Detected object: Class dog, Probability 0.81, Box with normalised locations [Center: (0.22, 0.72) Width: 0.10 Height: 0.14]
Box in original image size pixel locations: Box [Center: (140.49, 420.26) Width: 62.31 Height: 61.36]
Detected object: Class dog, Probability 0.89, Box with normalised locations [Center: (0.22, 0.73) Width: 0.19 Height: 0.20]
Box in original image size pixel locations: Box [Center: (141.57, 422.24) Width: 121.81 Height: 86.42]
Detected object: Class person, Probability 0.98, Box with normalised locations [Center: (0.36, 0.56) Width: 0.15 Height: 0.68]
Box in original image size pixel locations: Box [Center: (229.56, 352.94) Width: 98.66 Height: 286.97]

Timing statistics:
Network loading time (ms): 4712.62
Image loading time (ms): 597.75
Class names loading time: (ms) 1.27
Prediction time (ms): 196712.35
post_processing time (ms): 9.66 seconds
Total time: (ms) 202033.65

## NPU

Model Path: yolov5s-640-640.rknn
Input Path: image_640_640.jpg
Class Path: coco_80_labels_list.txt
Loop Count: 1
Initialised Model Successfully
rknn_api/rknnrt version: 1.4.1b9 (09eb4be80@2022-10-19T09:51:39), driver version: 0.8.2
model input num: 1, output num: 3
input tensors:
index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
output tensors:
index=0, name=output, n_dims=4, dims=[1, 80, 80, 255], n_elems=1632000, size=1632000, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003860
index=1, name=283, n_dims=4, dims=[1, 40, 40, 255], n_elems=408000, size=408000, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
index=2, name=285, n_dims=4, dims=[1, 20, 20, 255], n_elems=102000, size=102000, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003915
custom string:
Our model requires NHWC format with height=640,width=640,channels=3
Our image loaded with height=640,width=640,channels=3
Loaded input data from path : image_640_640.jpg
Creating tensor input memory
Copying input data to input tensor memory: width =640, stride=640
Creating tensor output memory
Setting tensor input memory
Setting tensor output memory
Begin perf ...
0: Elapse Time = 79.97ms, FPS = 12.50
model is NHWC input format
Post processing data
Confidence Threshold: 0.250000
Non-Maximum Suppression (NMS) Threshold: 0.250000
Scale Width: : 1.000000 Scale Height 1.000000
Loading labels
loadLabelName coco_80_labels_list.txt
Valid Count for Stride 8 : 0
Valid Count for Stride 16 : 14
Valid Count for Stride 32 : 22
result 0: ( 187, 161, 273, 563), person
result 1: ( 402, 213, 601, 520), horse
result 4: ( 66, 400, 203, 524), dog
person @ (187 161 273 563) 0.895416
horse @ (402 213 601 520) 0.872058
dog @ (66 400 203 524) 0.864517
Releasing rknn memory

Timing statistics:
Network loading time (ms): 4.93
Image loading time (ms): 262.38
Class names loading time: (ms) 0.44
Prediction time (ms): 83.08
Post Processing (ms): 11.37
Total time: (ms) 361.76

| Timing Statistics        | NPU (ms)   | Software Only (ms) |
| ------------------------ | ---------- | ------------------ |
| Network loading time     | 4.93       | 4712.62            |
| Image loading time       | 262.38     | 597.75             |
| Class names loading time | 0.44       | 1.27               |
| Prediction time          | 83.08      | 196712.35          |
| Post processing time     | 11.37      | 9.66               |
| **Total time**           | **361.76** | **202033.65**      |

| Timing Statistics | NPU (seconds) | Software Only (seconds) |
| ----------------- | ------------- | ----------------------- |
| **Total time**    | **0.36176**   | **202.03365**           |

Prediction time: 2368 times faster with NPU during prediction
Network loading time: 956 times faster with NPU

Timing statistics:
Network loading time (ms): 3153.33
Image loading time (ms): 591.00
Class names loading time: (ms) 0.19
Prediction time (ms): 196868.38
post_processing time (ms): 9.79 seconds
Total time: (ms) 200622.70
