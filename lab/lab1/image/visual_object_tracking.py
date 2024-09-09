import cv2
import numpy as np
import time

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# Initialize camera capture
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define the filename to save the pose data
filename = 'marker_poses.npz'

# Define an empty list to store the time series data
marker_poses = []

# Get the start time
start_time = time.time()

# Load the result from camera calibration
data = np.load('camera_params.npz')
mtx = data['mtx']
dist = data['dist']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the current time
    current_time = time.time()

    # Detect ArUco markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Loop over all detected markers
        for i in range(len(ids)):
            # Estimate the pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, mtx, dist)

            # Draw axis on the marker
            cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)

            # Get the time difference from the start of recognition
            time_diff = current_time - start_time

            # Add the ID and pose data to the time series
            marker_poses.append({'id': ids[i][0], 'rvec': rvec, 'tvec': tvec, 'time': time_diff})

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the time series data to file
np.savez(filename, marker_poses=marker_poses)

# Release capture and destroy window
cap.release()
cv2.destroyAllWindows()