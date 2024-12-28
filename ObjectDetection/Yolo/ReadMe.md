## Object Detection

Object detection is a computer vision task that involves identifying and localizing objects in an image or video. Unlike classification, which only labels the content of an image, object detection provides both the class of the object (e.g., "cat," "car") and its location within the frame, typically represented as bounding boxes. For example:

`Detected object: Class person, Probability 0.98, Box with normalised locations [Center: (0.36, 0.56) Width: 0.15 Height: 0.68]`

Object Detection is computationally intensive and poses some key challenges: The algorithm must detect objects in complex scenes with varying sizes, shapes, and appearances. For many applications like autonomous driving, surveillance, or robotics it must achieve real-time performance for applications like. It must also handle objects of different scales within a single image.

Over the years, various object detection approaches have emerged. One os the best known approach is Region-based Convolutional Neural Networks (R-CNN). This approach divides the object detection task into two stages: The first stage generates possible regions in the image where objects might exist. The second stage then uses a CNN to classify and refine these regions.

Single Shot MultiBox Detector (SSD) performs object detection in a single stage, without a separate region proposal step. It uses a grid of default bounding boxes and applies convolutional filters to predict object classes and refine bounding box coordinates directly.

YOLO (You Only Look Once) is another single-stage detection framework that frames object detection as a regression problem, emphasizing speed and simplicity. It is the initial focus of investigation into Object Detection and its integration into embedded systems.

YOLO is a deep learning-based object detection framework introduced by Joseph Redmon et al. in 2016. Its defining characteristic is its ability to perform object detection in a single pass through the neural network, making it extremely fast.

## How YOLO Works

The image is divided into an SxS grid of cells (e.g., 13x13 for YOLOv3 with a 416x416 input image). Each cell is responsible for predicting objects whose center falls within it.

In YOLOv3, the typical downsampling factor is 32x. This means the input image dimensions (width and height) are divided by 32 to calculate the final grid size. So for an image size of 608 X 608 we would end up with a grid of 19x19 in the final prediction layer.

Each cell predicts B bounding boxes. A bounding box includes x,y: Coordinates of the box center relative to the cell and Width and height relative to the image dimensions. Each cell also has a confidence score to indicate the likelihood of an object being in the box and the accuracy of the box. Each cell also predicts C class probabilities for the object.

YOLO combines these outputs (bounding boxes + class probabilities) in a single forward pass of the network. This enables end-to-end training, unlike region-based methods that split the task into separate stages.

## Applications

YOLO's speed and low complexity has made it a polular choice for **Autonomous Vehicles** where real-time detection of pedestrians, vehicles, and traffic signs is critical. It is also widely used in **Security and Surveillance** applications where it is often integrated into cameras. It has also become popular in **Retail Analytics** where smart cameras customer behavior analysis through object detection in stores.

## Darknet

The official implementations of YOLOv1, YOLOv2, and YOLOv3 are provided as part of the Darknet framework. Wile Darknet can work without a GPU much of its core implementation has been written in CUDA allowing providing seamless GPU acceleration for efficient training and inference. Compared to other frameworks Yolo has minimal dependencies and simple configuration files; these features will benefit us when we try to run Yola on Edge devices with constrained resources.

## Darknet Embedded

Darknet Embedded is a fork of the main Darknet project that has been optimised for Embedded devices. It focuses on the C API for darknet and does not require GPU support. It is also compatible with small foot-print C libraries like uclibc and musl which are specifically designed for embedded systems.
It can be integrated with Buildroot and a number of sample configurations are provided for both libc and uclibc based configurations.
