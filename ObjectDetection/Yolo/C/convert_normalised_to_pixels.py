import sys

def convert_normalized_to_pixel(box, image_size):
    """
    Convert normalized box coordinates to pixel coordinates.
    
    Args:
        box (tuple): Normalized (x, y, w, h) bounding box.
        image_size (tuple): (width, height) of the image.
    
    Returns:
        dict: Dictionary with center, top-left, and bottom-right coordinates.
    """
    x_norm, y_norm, w_norm, h_norm = box
    img_width, img_height = image_size
    
    # Center coordinates
    x_center = int(x_norm * img_width)
    y_center = int(y_norm * img_height)
    w_pixel = int(w_norm * img_width)
    h_pixel = int(h_norm * img_height)
    
    # Top-left and bottom-right coordinates
    x_top_left = x_center - (w_pixel // 2)
    y_top_left = y_center - (h_pixel // 2)
    x_bottom_right = x_center + (w_pixel // 2)
    y_bottom_right = y_center + (h_pixel // 2)
    
    return {
        "center": (x_center, y_center, w_pixel, h_pixel),
        "top_left": (x_top_left, y_top_left),
        "bottom_right": (x_bottom_right, y_bottom_right)
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python program.py x,y,w,h,img_width,img_height")
        sys.exit(1)
    
    try:
        # Parse command-line arguments
        args = sys.argv[1].split(',')
        if len(args) != 6:
            raise ValueError("Invalid number of arguments. Expected 6 values.")
        
        x, y, w, h = map(float, args[:4])
        img_width, img_height = map(int, args[4:])
        
        # Perform conversion
        results = convert_normalized_to_pixel((x, y, w, h), (img_width, img_height))
        
        # Display results
        print("📏 Bounding Box Conversion Results:")
        print(f"Center-based (x, y, w, h): {results['center']}")
        print(f"Top-Left Corner (x, y): {results['top_left']}")
        print(f"Bottom-Right Corner (x, y): {results['bottom_right']}")
    
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
