import cv2
import signal
import sys

# Graceful exit handler
def signal_handler(sig, frame):
    print("Interrupt received. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    sys.exit()

# Capture one frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture image from webcam. Exiting...")
    cap.release()
    sys.exit()

# Release the camera immediately after capturing the frame
cap.release()

# Display the captured frame
cv2.imshow("Captured Frame", frame)

print("Press any key to close the window...")

# Wait for a key press or window close event
cv2.waitKey(0)

# Clean up and close the program
cv2.destroyAllWindows()
print("Program exited cleanly.")
