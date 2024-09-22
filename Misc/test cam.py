import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to display frame using matplotlib
def display_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not video_capture.isOpened():
    print("Error: Webcam not found or unable to access.")
    exit()

plt.ion()  # Turn on interactive mode for matplotlib

# Start video stream
while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Frame not captured.")
        break

    # Display the frame using matplotlib
    display_frame(frame)

    # Exit the loop if 'q' is pressed
    if plt.waitforbuttonpress(0.01) and plt.get_current_fig_manager().canvas.get_default_filename().lower() == 'q':
        break

# Release webcam
video_capture.release()
plt.ioff()  # Turn off interactive mode
plt.close()  # Close the matplotlib window
