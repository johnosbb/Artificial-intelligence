import cv2
import csv

# Constants
IMAGE_PATH = '../data/image_416_416_13x13_grid.png'  # Path to the input image
CSV_PATH = '../data/raw_predictions_layer_16.csv'  # Path to the CSV file
OUTPUT_IMAGE_PATH = '../data/image_416_416_13x13_grid_overlayed.png'  # Path for the output image
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416

# Threshold for displaying bounding boxes
OBJECTNESS_THRESHOLD = 0.6  # Adjust this value as needed

def draw_predictions(image, csv_path, threshold):
    """
    Draw bounding boxes and objectness scores on the image based on CSV data.
    
    Args:
        image (ndarray): Loaded image using OpenCV.
        csv_path (str): Path to the CSV file containing predictions.
        threshold (float): Objectness score threshold for displaying predictions.
    """
    # Read CSV and parse bounding box data
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Extract values from the row
                anchor = row['anchor_number']
                objectness = float(row['objectness'])
                
                # Apply threshold
                if objectness < threshold:
                    continue  # Skip boxes below the threshold
                
                px = float(row['px'])
                py = float(row['py'])
                pw = float(row['pw'])
                ph = float(row['ph'])
                
                # Calculate bounding box corners
                x1 = int(px - (pw / 2))
                y1 = int(py - (ph / 2))
                x2 = int(px + (pw / 2))
                y2 = int(py + (ph / 2))
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color for the box
                thickness = 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # Place objectness score at the top-left corner of the box
                label = f"{anchor}: {objectness:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4  # Reduced font scale
                font_thickness = 1  # Thinner text for clarity
                
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x, text_y = x1, max(y1 - 5, 10)  # Ensure text stays within image bounds
                
                # # Draw background rectangle for text
                # cv2.rectangle(image, 
                #               (text_x, text_y - text_size[1]), 
                #               (text_x + text_size[0], text_y + 5), 
                #               (0, 255, 0), 
                #               -1)
                
                # Draw the text
                cv2.putText(image, 
                            label, 
                            (text_x, text_y), 
                            font, 
                            font_scale, 
                            (0, 0, 0), 
                            font_thickness)
                
            except KeyError as e:
                print(f"Missing key in CSV row: {e}")
            except ValueError as e:
                print(f"Value error: {e}")
    
    return image

def main():
    # Load the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    
    # Resize to expected dimensions (if necessary)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    # Draw predictions with threshold filtering
    annotated_image = draw_predictions(image, CSV_PATH, OBJECTNESS_THRESHOLD)
    
    # Save and display the output image
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_image)
    print(f"Overlayed image saved as {OUTPUT_IMAGE_PATH}")
    
    # Display the image
    cv2.imshow("Overlayed Predictions", annotated_image)
    
    # Wait for key press and ensure proper cleanup
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Wait until 'q' is pressed
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
