Converting YOLO Coordinates
To transform YOLO coordinates to pixel coordinates:

Multiply the normalized values by the dimensions of the image to get the pixel values.

Center X in pixels:
center_x = 0.36 × 640 = 230.4

Center Y in pixels:
center_y = 0.57 × 424 = 241.68

Width in pixels:  
width = 0.14 × 640 = 89.6

Height in pixels:
height = 0.72 × 424 = 305.28

In the case of cv2 example, x,y is the top left location of the bounding box for the object. The box has a width of 98 pixels and a height of 288 pixels.

For the Yolo example Box = x=0.36, y=0.57, width=0.14, height=0.72
x,y is the centre of the bounding box for the object. The box has a width of .14 and a height of .72 .
So the transformation of this into Converted Box in original image size (640x424): [231.10, 253.64, 88.58, 424.00]
