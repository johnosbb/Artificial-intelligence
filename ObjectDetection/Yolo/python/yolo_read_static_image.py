import cv2
import numpy as np

# Load YOLO model
config_path = 'yolov3.cfg'  # Path to YOLOv3 config file
weights_path = 'yolov3.weights'  # Path to YOLOv3 weights file
names_path = 'coco.names'  # Path to the file with class names

# Load class labels with error checking
try:
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        if not classes:
            raise ValueError("Class labels file is empty.")
except FileNotFoundError:
    print(f"Error: Class labels file '{names_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading class labels: {str(e)}")
    exit(1)

# Load YOLO network with error checking
try:
    net = cv2.dnn.readNet(weights_path, config_path)
    if net.empty():
        raise ValueError("Network model is empty. Check the weights and config file paths.")
except FileNotFoundError:
    print(f"Error: One or both of the files '{weights_path}' or '{config_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading YOLO network: {str(e)}")
    exit(1)

# Get layer names and output layers with error checking
try:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    if not output_layers:
        raise ValueError("No output layers found in the network.")
except Exception as e:
    print(f"Error processing network layers: {str(e)}")
    exit(1)

# Load the image
try:
    image = cv2.imread('image.jpg')
    if image is None:
        raise FileNotFoundError("Failed to load image - file not found or invalid image format")
    height, width, channels = image.shape
    if height == 0 or width == 0:
        raise ValueError("Invalid image dimensions")
        
except Exception as e:
    print(f"Error loading image: {str(e)}")
    exit(1)

# Prepare the image for YOLO (convert to blob)
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input for the network
net.setInput(blob)

# Forward pass through the network to get the output
outs = net.forward(output_layers)

# Process the output
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Filter weak detections
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
