import cv2 as cv
import numpy as np

# Open the video file
cap = cv.VideoCapture("happy.mp4")

# Read the first frame
ret, first_frame = cap.read()

# Resize the first frame
small_first_frame = cv.resize(first_frame, (640, 360))

# Convert to grayscale
prev_gray = cv.cvtColor(small_first_frame, cv.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(small_first_frame)
mask[..., 1] = 255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    small_frame = cv.resize(frame, (640, 360))

    # Show the input frame
    cv.imshow("input", small_frame)

    # Convert to grayscale
    gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of the flow vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Set the value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Show the dense optical flow
    cv.imshow("dense optical flow", rgb)

    # Update previous frame
    prev_gray = gray

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
