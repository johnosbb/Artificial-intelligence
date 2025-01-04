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

We can see from the results that this model is using diferent detection scales in the two main Yolo layers:

- Layer 9 (13×13 grid) for large objects.
- Layer 7 (26×26 grid) for medium-sized objects.

### Types of Layers

#### Convolutional Layer (conv)

The convolutional layer applies a set of filters (also called kernels) to an input feature map to create a new feature map. The operation involves sliding these filters over the input, performing element-wise multiplication between the filter and the section of the input it is currently covering, and summing the results to produce a single output value.

To extract features from the input (e.g., edges, textures, or more complex patterns). Each filter detects a different feature (e.g., horizontal lines, vertical lines, etc.), and the network learns which features are important during training.

The output of a convolutional layer is a feature map with typically higher depth (more channels) than the input, as each filter creates a new channel.

#### Max Pooling Layer (max)

A max pooling layer reduces the spatial dimensions of the input feature map by performing a downsampling operation. It works by dividing the input feature map into smaller regions (usually squares, such as $2 \times 2$) and picking the maximum value from each region. By downsampling, we reduce the spatial size (height and width) while preserving the most important features (by keeping the maximum value in each region). This helps reduce computational complexity and memory usage, as well as to make the network more invariant to small translations in the input (i.e., slight changes in the position of features).

There are two important parameters in this layer: The **Pool size**: is the size of the region from which the maximum value is taken (e.g., $2 \times 2$) and the **Stride**: How much the pooling window moves after each operation. A stride of 2 means it moves by 2 pixels after each pooling operation.
The output feature map will have smaller height and width but the same depth as the input. For example, applying $2 \times 2$ max pooling with stride 2 to an input of size $208 \times 208 \times 32$ would reduce the size to $104 \times 104 \times 32$.

#### Types of Layers

| **Layer**                                        | **Purpose**                                                                      | **Operation**                                                                                                                                             | **Effect on Size**                                                                  | **Common Use**                                                                                     |
| ------------------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Convolutional Layer** (`conv`)                 | Extract features (edges, textures, etc.)                                         | Applies filters (kernels) to the input to generate new feature maps. Each filter detects a different feature.                                             | Changes depth (channels); spatial size depends on stride and padding.               | Feature extraction from raw data (e.g., images).                                                   |
| **Max Pooling Layer** (`max`)                    | Downsample input, reducing spatial dimensions while retaining important features | Divides input into small regions (e.g., $2 \times 2$) and outputs the maximum value in each region.                                                       | Reduces spatial dimensions (height and width), keeps depth.                         | Reduces computational load, prevents overfitting, and introduces spatial invariance.               |
| **Average Pooling Layer** (`avg`)                | Downsample input while keeping average values of regions                         | Divides input into regions (e.g., $2 \times 2$) and outputs the average value of each region.                                                             | Reduces spatial dimensions (height and width), keeps depth.                         | Reduces feature map size, maintains important spatial information.                                 |
| **Fully Connected Layer** (`fc`)                 | Combine features extracted from the convolutional layers                         | Connects every neuron in the layer to every neuron in the next layer.                                                                                     | Reduces the dimensionality and produces a final output layer.                       | Final decision-making layer (e.g., classification or regression).                                  |
| **Batch Normalization Layer** (`bn`)             | Normalize activations to stabilize training                                      | Normalizes the activations of the neurons in a layer, ensuring that they are centered around zero and have unit variance.                                 | No change to size (spatial or depth); changes activations.                          | Improves convergence, speeds up training, and reduces overfitting.                                 |
| **Activation Layer** (`ReLU`, `sigmoid`, `tanh`) | Introduce non-linearity to the model                                             | Applies a non-linear activation function like ReLU (Rectified Linear Unit), sigmoid, or tanh element-wise to the input.                                   | No change to size (spatial or depth); changes activations.                          | Adds non-linearity to the model, enabling it to learn complex patterns.                            |
| **Dropout Layer** (`dropout`)                    | Prevent overfitting by randomly deactivating neurons during training             | Randomly sets a fraction of the input units to zero during training (usually between 0.2 to 0.5).                                                         | No change to size; reduces the number of active neurons.                            | Regularization technique to reduce overfitting.                                                    |
| **Flatten Layer**                                | Flatten multi-dimensional inputs into a 1D vector for the fully connected layer  | Converts the multi-dimensional input (e.g., a 3D feature map) into a 1D vector that can be passed to fully connected layers.                              | Reduces dimensions to a 1D vector (e.g., $N \times H \times W$ to $N$)              | Required before passing data to fully connected layers.                                            |
| **Up-sampling Layer** (`upsample`)               | Increase spatial dimensions of the feature map                                   | Increases the size of the input by duplicating or interpolating values.                                                                                   | Increases spatial dimensions (height and width).                                    | Used in architectures like autoencoders or generative models to produce higher-resolution outputs. |
| **Yolo Layer** (`yolo`)                          | Perform object detection and bounding box prediction                             | Predicts class probabilities, objectness scores, and bounding box coordinates. Each grid cell in the output layer is responsible for detecting an object. | Output grid containing predictions (e.g., class probabilities and box coordinates). | Used in object detection networks like YOLO to output class and bounding box predictions.          |

_summary of different layer types_

#### Layers used in the Yolo Model

```
Layer    Filters    Size/Strides  Input Dimension      Output Dimension    BFLOPS
0   conv      16       3x3/1       416x416x3        ->  416x416x16        0.150 BF
1   max                2x2/2       416x416x16       ->  208x208x16        0.003 BF
2   conv      32       3x3/1       208x208x16       ->  208x208x32        0.399 BF
3   max                2x2/2       208x208x32       ->  104x104x32        0.001 BF
4   conv      64       3x3/1       104x104x32       ->  104x104x64        0.399 BF
5   max                2x2/2       104x104x64       ->  52x52x64          0.001 BF
6   conv      128      3x3/1       52x52x64         ->  52x52x128         0.399 BF
7   max                2x2/2       52x52x128        ->  26x26x128         0.000 BF
8   conv      256      3x3/1       26x26x128        ->  26x26x256         0.399 BF
9   max                2x2/2       26x26x256        ->  13x13x256         0.000 BF
10  conv      512      3x3/1       13x13x256        ->  13x13x512         0.399 BF
11  max                2x2/1       13x13x512        ->  13x13x512         0.000 BF
12  conv      1024     3x3/1       13x13x512        ->  13x13x1024        1.595 BF
13  conv      256      1x1/1       13x13x1024       ->  13x13x256         0.089 BF
14  conv      512      3x3/1       13x13x256        ->  13x13x512         0.399 BF
15  conv      255      1x1/1       13x13x512        ->  13x13x255         0.044 BF
16  yolo                -           13x13x255       ->  -                  -
17  route               -           13               ->  13x13x256         -
18  conv      128      1x1/1       13x13x256        ->  13x13x128         0.011 BF
19  upsample            -           13x13x128        ->  26x26x128         -
20  route               19 8        -                 ->  26x26x384         -
21  conv      256      3x3/1       26x26x384        ->  26x26x256         1.196 BF
22  conv      255      1x1/1       26x26x256        ->  26x26x255         0.088 BF
23  yolo                -           26x26x255       ->  -                  -
```

#### Layer Breakdown for our Yolo Model

- Convolutional Layers (conv)

  - Layers: 0, 2, 4, 6, 8, 10, 12, 13, 14, 21
  - Purpose: Feature extraction by applying convolutional filters. These layers are crucial for detecting patterns like edges, textures, and more complex features at deeper layers.
  - Total: 10 layers (conv layers).

- Max Pooling Layers (max)

  - Layers: 1, 3, 5, 7, 9, 11
  - Purpose: Downsample the feature map by taking the maximum value in each local region. This reduces the spatial dimensions (height and width) while retaining the most important information.
  - Total: 6 layers (max pooling).

- YOLO Layer (yolo)

  - Layers: 16, 23
  - Purpose: Perform object detection by predicting class labels, bounding box coordinates, and confidence scores. The output consists of class probabilities and bounding boxes for each grid cell.
  - Total: 2 layers (YOLO layers).

- Route Layer (route)

  - Layers: 17, 20
  - Purpose: This layer concatenates feature maps from different parts of the network to provide multi-scale feature information for later stages, such as detection.
  - Total: 2 layers (route layers).

- Upsample Layer (upsample)
  - Layer: 19
  - Purpose: Upsample the feature map, increasing its spatial resolution, which is especially useful for models like YOLO that need high-resolution feature maps for detection.
  - Total: 1 layer (upsample layer).
