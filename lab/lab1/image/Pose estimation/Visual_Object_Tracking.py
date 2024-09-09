import cv2
import numpy as np
import time
import cv2.aruco as aruco

# Define ArUco dictionary and parameters
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
arucoParams = aruco.DetectorParameters_create()

# Initialize camera capture
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if ids is not None:
        # Loop over all detected markers
        for i in range(len(ids)):
            # Estimate the pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 20, mtx, dist)

            # Draw axis on the marker
            cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 10)

            # Get the time difference from the start of recognition
            time_diff = current_time - start_time

            # Add the ID and pose data to the time series
            marker_poses.append({'id': ids[i][0], 'rvec': rvec, 'tvec': tvec, 'time': time_diff})

            # Compute homogenous transformation matrix
            rmat = cv2.Rodrigues(rvec)[0]
            homogenous_trans_mtx = np.append(rmat, [[tvec[0][0][0]], [tvec[0][0][1]], [tvec[0][0][2]]], axis=1)
            homogenous_trans_mtx = np.append(homogenous_trans_mtx, [[0, 0, 0, 1]], axis=0)
            print('id: ', ids[i], 'time:', round(current_time - start_time, 3))
            print("homogenous_trans_matrix\n", np.array2string(homogenous_trans_mtx, precision=3, suppress_small=True))

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the time series data to file
filename = 'marker_poses.npz'
np.savez(filename, marker_poses=marker_poses)

# Release capture and destroy window
cap.release()
cv2.destroyAllWindows()
