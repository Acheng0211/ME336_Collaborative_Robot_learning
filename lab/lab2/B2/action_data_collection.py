import cv2
import time
import numpy as np
import math
import cv2.aruco as aruco

if __name__ == '__main__':
    aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters_create()
    '''
    If an error occurs, please use the following two lines of code.
    如果报错，请将上面两行代码注释掉，用后面这两行代码
    '''
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # aruco_params = aruco.DetectorParameters()

    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

    # Initialize camera capture
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(cap.get(3), cap.get(4), cap.get(5))

    d23_min = 1
    d45_min = 1
    d23_max = 0
    d45_max = 0
    calibFlag = False
    # Define an empty list to store the time series data
    marker_poses = []

    # Get the start time
    start_time = time.time()

    # Load the result from camera calibration
    data = np.load('camera_params.npz')
    mtx = data['mtx']
    dist = data['dist']

    force_left_finger = 0
    force_right_finger = 0
    midpointTvec = np.array([0, 0, 0])
    trai_datas = []
    trai_data = np.array([0, 0])  # x,y for the midPoint for marker 0,1
    force_datas = []
    force_data = np.array([0, 0])

    # Play the video
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        if ret == True:
            # Get the current time
            current_time = time.time()
            # Detect ArUco markers in the frame
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            if ids is not None:
                # Loop over all detected markers
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
                for i in range(0, ids.size):
                    # draw axis for the aruco markers
                    cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)
                if 2 in ids and 3 in ids:
                    marker2_idx = np.where(ids == 2)[0][0]
                    marker3_idx = np.where(ids == 3)[0][0]

                    t2 = tvecs[marker2_idx].reshape(-1)
                    t3 = tvecs[marker3_idx].reshape(-1)
                    d23 = math.sqrt((t2[0] - t3[0]) ** 2 + (t2[1] - t3[1]) ** 2)
                    if d23 < d23_min:
                        d23_min = d23
                        force_left_finger = 0
                    elif d23 > d23_max:
                        d23_max = d23
                        force_left_finger = 1
                    else:
                        force_left_finger = (d23 - d23_min) / (d23_max - d23_min)

                if 4 in ids and 5 in ids:
                    marker4_idx = np.where(ids == 4)[0][0]
                    marker5_idx = np.where(ids == 5)[0][0]

                    t4 = tvecs[marker4_idx].reshape(-1)
                    t5 = tvecs[marker5_idx].reshape(-1)
                    d45 = math.sqrt((t4[0] - t5[0]) ** 2 + (t4[1] - t5[1]) ** 2)
                    if d45 < d45_min:
                        d45_min = d45
                        force_right_finger = 0
                    elif d45 > d45_max:
                        d45_max = d45
                        force_right_finger = 1
                    else:
                        force_right_finger = (d45 - d45_min) / (d45_max - d45_min)

                if 3 in ids and 5 in ids:
                    marker3_idx = np.where(ids == 3)[0][0]
                    marker5_idx = np.where(ids == 5)[0][0]
                    midpointTvec = ((tvecs[marker3_idx] + tvecs[marker5_idx]) / 2)[0]
                    # print(midpointTvec)

                if 2 in ids and 3 in ids and 4 in ids and 5 in ids:
                    print("Force:", force_left_finger, force_right_finger)
                    if calibFlag:
                        traj_data = np.array([midpointTvec[0], midpointTvec[1]])
                        trai_datas.append(traj_data)
                        force_data = np.array([force_left_finger, force_right_finger])
                        force_datas.append(force_data)

            if True:
                # Display the frame
                cv2.imshow('Frame', frame)
                # Exit on key press
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    calibFlag = True
                    print("Start Collection...")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Wait for the specified delay (in milliseconds)
                cv2.waitKey(1)
            else:
                break
    print(len(trai_datas))
    print(len(force_datas))

    np.save('trajdataForLFD_acheng.npy', trai_datas)
    np.save('forcedataForLFD_acheng.npy', force_datas)

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

