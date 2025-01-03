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

## In-depth Analysis

### First Layer

```
 0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
```

After the first layer we are left with an output of 416 x 416 x 16. The input representing the image 416 x 416 x 3 (the 3 is the colour components R,G and B) goes through a convolutional layer with 16 filters. Each filter extracts a specific feature map from the image (e.g., edges, textures, patterns) producing of 16 feature maps. Each of these 16 channels represents a different feature map that was detected by one of the 16 convolutional filters. A convolutional filter (e.g., 3×3×3 for RGB input) slides over the image and computes a dot product at each position, producing a single channel output per filter, so with 16 filters, you get 16 channels in the output. We can also see in this output that this process took 0.150 BF, which means this convolutional layer requires 0.150 billion floating-point operations to process the input.

A kernel (or filter) is a small matrix used to extract specific features from an image, such as edges or textures. In our case we can calculate the billion floating-point operations using the following formula:

$FLOPs = 2 \cdot (k^2) \cdot C_{in} \cdot C_{out} \cdot W \cdot H$

where k is kernel size (3 x 3) and C is the number of channels, and W, H is the width and height. This fives us:

$= 2 \cdot 9 \cdot 3 \cdot 16 \cdot 416 \cdot 416$
$= 149,520,384$

and as:

$1 \text{ BFLOP} = 10^9 \text{ FLOPs}$

We have a final calculation of:

$BFLOPS = \frac{149,520,384}{1,000,000,000} = 0.150 \text{ BFLOPS}$.

### Second Layer

```
   1 max                2x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16 0.003 BF
```

The second layer is responsible for downsizing the image feature maps, so the width and height of the feature map are reduced by a factor of 2 (e.g., from $416 \times 416$ to $208 \times 208$). This reduces the computational cost and memory usage in subsequent layers. Smaller feature maps mean fewer calculations and less data to process. Each pixel in the downsampled feature map corresponds to a larger region in the original image and this allows the network to "see" larger parts of the image at higher levels, which helps in understanding more abstract and global features. Lower-level layers capture fine details, while downsampled layers focus on more abstract, high-level features. Object detection benefits from both local fine-grained details (e.g., edges, textures) and higher-level features (e.g., shapes, patterns).

YOLO predicts objects at multiple scales. Downsampling creates multi-scale feature maps that enable the network to detect both small and large objects effectively. Smaller objects might be detected in earlier layers (finer resolution), while larger objects are detected in later layers (coarser resolution). Reducing the dimensions reduces the number of parameters in subsequent layers and this process also helps prevent overfitting, especially when training with limited data.

### Third Layer

```
   2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32 0.399 BF
```

The third layer performs convolution with 32 filters of size $3 \times 3$. Each filter is applied to the input feature map (of size 208 x 208 x 16), creating 32 different feature maps. This layer extracts different features from the previous layer’s outputs, such as edges, textures, or patterns. While the spatial dimensions (height and width) of the feature map remain the same (208 x 208), the depth increases from 16 to 32, indicating that the network is learning to represent more complex features by adding additional channels. After passing through earlier layers, the network is beginning to capture higher-level features, such as combinations of basic edges and textures. The third convolution layer allows for more sophisticated abstraction by increasing the number of features extracted from the image.

### Intermediate Layers

This process of convulution and Pooling continues for a number of layers.

### Layer 16

YOLO uses a technique called anchor boxes (or prior boxes) to help guide the model in predicting bounding box shapes. Anchor boxes are predefined bounding boxes with fixed aspect ratios and sizes, which represent typical object shapes.

In Yolo these anchors are defined in the cfg file

```
[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

The mask specifies which anchors are used at a particular YOLO layer.
In this example the mask is 3,4,5 so only anchors 3, 4, and 5 (81,82, 135,169, 344,319) will be used at this detection layer.
Different YOLO layers often focus on detecting objects at different scales.

In Layer 16 (with a 13×13 grid), the model assigns three anchor boxes to each grid cell. At this lower resolution, these anchor boxes are primarily responsible for detecting larger objects in the image.

For each anchor box, YOLO calculates:

An objectness score, which estimates the likelihood that the box contains any object.
Class probabilities, which estimate the likelihood that the detected object belongs to one of the predefined classes specified in the coco.names file.
Together, these scores help YOLO determine what the object is and where it is located in the image.

### Layer 23

In layer 23 a 26×26 grid is used. This grid size is used to smaller objects, as well as the class probabilities and objectness scores for each of those boxes.

The process is similar to layer 16, the model assigns three anchor boxes to each grid cell and then again calculates an objectness score and class probabilities for each of the anchor boxes. Again, the cfg file defines the anchor box dimensions and the mask indicates for this layer we should use: `10,14,  23,27,  37,58`

```
[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

### Example: How YOLO Uses Bounding Boxes and Objectness Scores

- Suppose we have an image of size **416×416** and a **13×13 grid** (optimized for detecting **large objects**).
- The YOLO model divides the image into **13×13 grid cells**.
- Each **grid cell** predicts **3 bounding boxes**, where each bounding box includes:
  - **Coordinates**: _(x, y, w, h)_ representing the center (x, y), width (w), and height (h) of the box.
  - An **objectness score**: Represents the **confidence** that the box contains any object.
  - **Class probabilities**: A probability distribution across all object classes (e.g., _dog_, _car_, etc.).

For a **single grid cell**, YOLO might predict:

- **Box 1:**

  - **Coordinates:** _(x=0.2, y=0.3, w=0.4, h=0.5)_
  - **Objectness Score:** **0.8** _(80% confident that an object is present)_
  - **Class Probabilities:** _(dog=0.9, car=0.1, ...)_ — likely a **dog**.

- **Box 2:**

  - **Coordinates:** _(x=0.6, y=0.4, w=0.3, h=0.2)_
  - **Objectness Score:** **0.4** _(40% confident that an object is present)_
  - **Class Probabilities:** _(dog=0.3, car=0.7, ...)_ — likely a **car**.

- **Box 3:**

  - **Coordinates:** _(x=0.8, y=0.7, w=0.2, h=0.2)_
  - **Objectness Score:** **0.9** _(90% confident that an object is present)_
  - **Class Probabilities:** _(dog=0.9, car=0.1, ...)_ — likely a **dog**.

The network starts with a set of anchor boxes of a predefined size: (w anchor ,h anchor ). These anchors act as reference shapes for potential objects in the image. For each of these boxes the _network_ predicts offsets: (ŵ, ĥ). These offsets (which can be psoitive or negative) are used to further refine the anchor boxes dimensions to more closely match the actual object dimensions. The predicted x and y values are normalized offsets (ranging between 0 and 1) relative to the top-left corner of the grid cell. The predicted offsets are passed through an exponential function (e^(ŵ), e^(ĥ)) to ensure the final width and height remain positive.

The final center coordinates of the bounding box are computed using the offsets and the grid cell location:

#### Final Bounding Box Center Coordinates

The final center coordinates $(x_{\text{final}}, y_{\text{final}})$ of the predicted bounding box are calculated using the grid cell location, normalized offsets $(x)$ and $(y)$, and the size of each grid cell.

**Formulas:**

- $x_{\text{final}} = (grid\_x + x) \times cell\_width$
- $y_{\text{final}} = (grid\_y + y) \times cell\_height$

**Where:**

- $grid\_x$: The x-coordinate of the grid cell.
- $grid\_y$: The y-coordinate of the grid cell.
- $x$: Normalized horizontal offset (e.g., 0.62).
- $y$: Normalized vertical offset (e.g., 0.49).
- $cell\_width$: The width of a grid cell.
- $cell\_height$: The height of a grid cell.

We can calculate the final width and height as follows:

- $w_{\text{final}} = w_{\text{anchor}} \times e^{\hat{w}}$
- $h_{\text{final}} = h_{\text{anchor}} \times e^{\hat{h}}$

If the centre of a bounding box falls within a cell then that cell is responsible for that bounding box. These refined bounding boxes (with accurate width and height after transformation) are then scored for objectness (how confident the model is about an object being present) and for class probabilities (which class the object belongs to).

### Objectness

The objectness score is a scalar value (ranging from 0 to 1) predicted for each anchor box. It represents the model's confidence that an object exists within that specific anchor box. If the objectness score exceeds a predefined threshold (e.g., 0.5), the model considers the anchor box to contain an object.

Objectness Score ($P_{\text{obj}}$)

### Class Probability

For each anchor box where the objectness score is high enough, YOLO also predicts a class probability distribution over all possible object classes. The class with the highest probability is used to determine the predicted class.

Class Probabilities ($P_{\text{class}_i}$)

### Candidate Bounding Boxes

This process results in a list of candidate bounding boxes (sometimes called detection boxes) with an objectness score and a set of class probability scores. These bounding boxes represent the model's best guess for object locations and dimensions. To remove redundant and overlapping boxes, Non-Maximum Suppression (NMS) is applied. NMS ensures that only the highest-confidence box for each object (e.g., Box 3) is kept, while overlapping lower-confidence boxes are discarded.

Each bounding box then will have:

Center Coordinates ($x$, $y$) – Offset within the grid cell.
Width and Height ($w$, $h$) – Refined bounding box dimensions.
Objectness Score ($P_{\text{obj}}$) – Probability of an object existing.
Class Probabilities ($P(\text{class}_i | \text{object})$) – Probabilities for each class.

This means YOLO doesn’t just detect if there’s an object—it also predicts what the object is in a single forward pass of the network.

### Objectness Threshold in YOLO

The threshold is a predefined value used to determine whether an anchor box contains an object or not, based on the objectness score ($P_{\text{obj}}$).
Typical Threshold Values
In most YOLO implementations:

Default Objectness Threshold: $0.5$
This means if $P_{\text{obj}} > 0.5$, the model considers the anchor box to contain an object.
The threshold can be adjusted based on the application:

Higher Threshold (e.g., 0.7): Reduces false positives but might miss some valid detections (false negatives).
Lower Threshold (e.g., 0.3): Increases the number of detected objects but might introduce more false positives.

### Selecting based on Objectness

The objectness score is a probability between 0 and 1 that indicates the confidence level of the model that an object exists in a specific anchor box. If the objectness score exceeds the threshold, the anchor box is considered to contain an object. If it falls below the threshold, the box is ignored during the next stages of processing.

IoU= Area of Intersection/Area of Union
​

Intersection: The area where the two bounding boxes overlap.
Union: The total area covered by both bounding boxes, minus the overlap.
Measures how well a predicted bounding box matches a ground-truth box.
Thresholding: Typically, an IoU threshold (e.g., 0.5) is used to determine if a detection is valid.

Area of A + Area of B - Intersection
We see that the portion of the intersection is covered in both the boxes. Since we want to account for the common area of intersection only once, we can subtract the area of intersection we calculated, from the total area of the two boxes.

The code processes one class at a time (k).
Inside this loop:
The bounding boxes (a and b) are compared using box_iou.
If two boxes have a high IoU (> thresh) and are for the same class (prob[k]), the less confident box (dets[j].prob[k]) is suppressed.

anchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319
Mask specifies which anchors are used at this YOLO layer.
In this example the mask is 3,4,5 so only anchors 3, 4, and 5 (81,82, 135,169, 344,319) will be used at this detection layer.
Different YOLO layers often focus on detecting objects at different scales.

nms sorting

The goal is to suppress overlapping boxes that are less confident while keeping the most confident bounding box for each class.
Sorting First: Ensures the highest-confidence box is evaluated first.
Boxes are suppressed if they overlap too much with a higher-confidence box.
After this process, the remaining boxes for each class are non-overlapping and high-confidence detections.

When we perform Non-Maximum Suppression (NMS), the goal is to remove redundant bounding boxes that predict the same object. In the comparison box_iou(a, b), we compare two detection boxes (a and b) for the same class and check their Intersection over Union (IoU) score.

If the IoU exceeds a given threshold (thresh), box b is suppressed by setting its probability to 0.

The IoU (Intersection over Union) check in Non-Maximum Suppression (NMS) is used to determine whether two bounding boxes (a and b) are likely referring to the same object in the image.

An IoU score close to 1.0 means the two boxes overlap almost perfectly, suggesting they are likely detecting the same object.
An IoU below a threshold (e.g., 0.5) suggests they might represent different objects.

IoU is a spatial overlap metric used to identify redundant boxes predicting the same object.
A high IoU suggests significant overlap, indicating that two boxes are likely focused on the same object instance.
The box with the higher confidence score is retained, and the less confident one is suppressed.
