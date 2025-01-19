# Explanation

## The Object Detection Process

### Image Input

The process begins with an input image or video frame that the computer needs to analyze.

### Feature Extraction

The computer processes the image using a neural network, typically a Convolutional Neural Network (CNN). This network scans the image and extracts important features (such as edges, textures, and patterns) that help identify different objects.
These features get combined to form more complex shapes that represent parts of objects (e.g., wheels, faces, or letters).

### Grid System and Stride

The image is divided into a grid (like a checkerboard). Each grid cell is responsible for detecting whether an object exists in that section of the image and predicting the properties of the object. In object recognition systems like those used in YOLO (You Only Look Once) or similar models, the grid is typically a fixed grid that overlays the entire image.

While each grid is fixed over the image, using different grid sizes allows the model to detect objects of various sizes. The _stride_ determines the size of the cell within the fixed grid. The _stride_ determines the spacing between these grid cells in the original image. A stride of 32 in a 416x416 image results in grid cells that are spaced 32 pixels apart, creating a grid with fewer, larger cells.

- Small grid (coarse, fewer cells): Each cell covers a larger portion of the image and can detect larger objects.
- Medium grid: Balances between small and large object detection.
- Large grid (fine, more cells): Each cell covers a smaller area, which is better for detecting small objects.

Each of these grids (small, medium, and large) acts as a fixed grid over the image when considered individually. The different grid levels (with varying strides) overlay the image simultaneously during the detection process, allowing the model to capture objects at different scales effectively. This approach is crucial for ensuring that objects of different sizes—whether small, medium, or large—are detected reliably within the image.

| Stride | Image Size     | Grid Size |
| ------ | -------------- | --------- |
| 8      | 416x416 pixels | 52x52     |
| 16     | 416x416 pixels | 26x26     |
| 32     | 416x416 pixels | 13x13     |

### Window Size

The window size (also called kernel size) refers to the size of the filter or "receptive field" used in the convolution process to scan the image. This is the fixed size of the area that the model examines in each step. For example, a 3x3 window means the model looks at a 3x3 region of the image at a time, and a 5x5 window means it looks at a 5x5 region.

Window size (or kernel size) is typically pre-defined when designing the model architecture and can vary depending on the use case and the type of features you want the network to learn. In common convolutional neural networks (CNNs) and object detection models (like YOLO), typical window sizes (kernel sizes) can be:

- 3x3: A very common size, used in many layers to capture fine details.
- 5x5 or 7x7: Used for capturing larger patterns or features.

Stride and window size work together to define the coverage of each convolutional operation. The window size determines the region of the image being examined in each step. The stride determines how far the window moves after each operation.

Assume we have a 7x7 image

```
[1 2 3 4 5 6 7]
[8 9 10 11 12 13 14]
[15 16 17 18 19 20 21]
[22 23 24 25 26 27 28]
[29 30 31 32 33 34 35]
[36 37 38 39 40 41 42]
[43 44 45 46 47 48 49]
```

With a 5x5 window and stride 2, the model would examine regions of the image like this:

First step (top-left corner):

```
[1 2 3 4 5]
[8 9 10 11 12]
[15 16 17 18 19]
[22 23 24 25 26]
[29 30 31 32 33]
```

Second step (shifted 2 pixels to the right):

```
[3 4 5 6 7]
[10 11 12 13 14]
[17 18 19 20 21]
[24 25 26 27 28]
[31 32 33 34 35]
```

Third step (shifted 2 pixels down, back to the left side):

```
[15 16 17 18 19]
[22 23 24 25 26]
[29 30 31 32 33]
[36 37 38 39 40]
[43 44 45 46 47]
```

So, with this 5x5 window and a stride of 2, the model would slide over the image and examine a 5x5 section at each step, but move by 2 pixels after each operation.

In summary then: window size (kernel size) determines the size of the region examined by the model at each step. It's typically fixed and chosen based on the model architecture and stride determines how far the window moves after each operation, affecting the size of the output grid and the level of detail captured.

### Anchor Boxes

The system uses predefined "anchor boxes" of various shapes and sizes. These boxes help predict different types of objects (e.g., a tall box for a person, a wide box for a car).
Each grid cell tries to match these anchor boxes to objects it detects.

### Classification and Bounding Box Prediction

The network outputs predictions for each anchor box:

- Bounding box coordinates (to draw a box around the detected object).
- Confidence score (a measure of how likely it is that an object is in the box).
- Class probability (e.g., "cat," "dog," or "car").

If the confidence score is high enough, the system considers it a potential detection.

### Non-Max Suppression

Sometimes, multiple boxes may overlap around the same object. To avoid duplicate detections, the system uses a method called non-max suppression. This step keeps only the box with the highest confidence score and removes the others.

### What Happens When an Object Spans Multiple Cells

If an object, such as a bus, spans across multiple grid cells (e.g., half in grid cell (0,0) and the other half in (0,1)), each cell will process its portion of the object independently.
Typically, only one cell (often the one containing the center of the object) is responsible for predicting the full bounding box for that object.

Each grid cell will predict one or more bounding boxes, complete with coordinates, confidence scores, and class labels.
If a part of the object is in multiple grid cells, multiple cells might initially make predictions.

The model evaluates the confidence scores for each predicted bounding box. To ensure only one box is reported per object, non-max suppression is applied to eliminate overlapping boxes and keep the one with the highest confidence score.

Suppose a bus is located at the boundary between grid cells (0,0) and (0,1); the cell containing the center of the bus (let's say it's in (0,1)) would be primarily responsible for predicting the full bounding box for the bus. Grid cell (0,0) might detect parts of the bus but won't generate a strong prediction for it because it doesn't contain the center. The model uses non-max suppression to combine overlapping detections and keep only the most confident prediction.

### Final Output

The system outputs the detected objects, including the bounding box coordinates, confidence scores, and class labels.

### The Requirement to Scale the Bounding Boxes

In object detection, scaling up and down is a critical part of adjusting the coordinates and dimensions of detected bounding boxes to match the original image size or the model's input dimensions.

#### Image Preprocessing and Model Input Size

Before passing an image to an object detection model, it is often resized to a standard input size (e.g., 416x416 pixels). This resizing ensures consistency, so the model always processes inputs of the same shape and can operate efficiently. During this step, the image and all the objects within it are proportionally scaled down to fit the model's expected input size.

Inside the model, the bounding box coordinates are often normalized to the scaled-down input space. This means that when the model predicts the location of an object, it uses coordinates relative to the resized version of the image (e.g., in the 416x416 space, not the original image's size).

To make the predictions usable in the context of the original image, the bounding box coordinates and sizes must be scaled back up to the original dimensions. For example, if the original image was 1280x720 and it was resized to 416x416 for the model, any detected bounding boxes need to be scaled up from the 416x416 space back to the 1280x720 space.

### Example

Imagine a photo of a street scene. The object recognition model would preform the following steps:

- Breaks the image into a grid and checks each section.
- Uses anchor boxes to look for objects like cars and pedestrians.
- Calculates the confidence for each detection and decides which boxes represent real objects.
- Outputs the locations of objects (e.g., "car at (50, 30), person at (100, 60)") with bounding boxes around them.

## Strides

In the context of object detection models, the term stride refers to how much the model's receptive field shifts over the input image during feature extraction. Strides are crucial in determining the granularity at which the model analyzes the image.

### Feature Map Resolution

Stride dictates the down-sampling rate between the input image and the generated feature map. A stride of 8, 16, or 32 means that the spatial resolution of the feature map is reduced by a factor of 8, 16, or 32, respectively, compared to the original image.
Lower strides (e.g., 8) produce larger feature maps with finer details, suitable for detecting smaller objects.
Higher strides (e.g., 32) produce smaller feature maps with less spatial detail, which are better suited for detecting larger objects.

### Multi-scale Detection

Many object detection models (e.g., YOLO, SSD) use different strides to create multiple feature maps at different scales. This allows the model to detect objects of various sizes more effectively.
The feature map with a stride of 8 will detect smaller objects as it covers the input image in finer detail.
The feature map with a stride of 16 or 32 will detect larger objects as it spans larger portions of the input image.

### Anchor Boxes and Detection

Each feature map generated at a specific stride is associated with a set of anchor boxes (or predefined bounding box templates) scaled appropriately for that stride. The model then predicts object presence and class probabilities relative to these anchor boxes.
For example, in the provided code:
The process_native_nhwc() function processes the feature maps at strides 8, 16, and 32.
These strides determine how densely the detection grid is laid out on the image.
A feature map with a stride of 8 will have more cells (higher resolution grid) compared to one with a stride of 32, resulting in different coverage areas for detecting objects. The choice of stride allows a trade-off between accuracy and speed.

- Lower strides (more detailed feature maps) typically yield better detection of small objects but require more computation.
- Higher strides (more abstract feature maps) require less computation but may miss smaller objects due to coarser coverage.
- Balancing multiple strides allows models to maintain accuracy across a range of object sizes while optimizing inference speed.

### Practical Implications

In this example I provided processes for three different feature maps, each with a different stride (8, 16, 32). This approach helps in detecting objects of varying scales:

- Stride 8 for small objects.
- Stride 16 for medium objects.
- Stride 32 for large objects.

## Labels

The purpose of the labels file in an object detection or classification system is to provide a mapping between the numerical class IDs predicted by the model and their corresponding human-readable class names.

The model outputs class IDs (integers) for detected objects, which represent the categories the model has learned to recognize. The labels file contains a list of class names corresponding to these IDs. For example, if the model outputs a class ID of 0, the labels file may map this to the name "person," and an ID of 1 might map to "bicycle."

### Readable Outputs

The labels file allows the program to translate model predictions from numerical values into descriptive names that are human-readable. This makes it easier for users to understand the results (e.g., displaying "bus" instead of ID 5).

## Standardized Input

The labels file typically follows a standard format (e.g., one class name per line), making it easy for functions like loadLabelName() to read and store the labels in an array or vector. This array can be used to look up and print the name of a detected class when populating result structures or displaying outputs.

### Labels File Example

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
```
