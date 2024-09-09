import cv2
import numpy as np
import os
import time
import shutil
import sys

# Set the chessboard size and grid width
num_horizontal = 11
num_vertical = 8
grid_width = 0.007  # meter


def capture_calibration_images(camera_id, num_images, delay=3):
    folder = 'cali_imgs'
    if os.path.exists(folder):
        # Delete all files in the folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # If the folder does not exist, create it
        os.makedirs(folder)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Failed to open the camera. Reason: {cap}")
        return

    photo_count = 0
    start_time = time.time()

    while cap.isOpened() and photo_count < num_images:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate the remaining time for the countdown (in seconds)
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = delay - elapsed_time if elapsed_time < delay else 0

        # Calculate the end angle of the sector
        end_angle = 360 * remaining_time / delay

        # Create a copy of the original image with the same size
        overlay = np.zeros_like(frame)

        # Create a copy of the original image with the same size
        overlay = frame.copy()

        # Draw a white ellipse on the copy
        cv2.ellipse(overlay, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)),
                    (int(frame.shape[0] * 0.5), int(frame.shape[0] * 0.5)), -90, 0, end_angle, (255, 255, 255), -1)

        # Blend the copy and the original image with a transparency of 50%
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, f'Photo {photo_count + 1}/{num_images}', (frame.shape[1] - 200, frame.shape[0] - 10), font,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Calibration', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the image every delay seconds
        if time.time() - start_time >= delay:
            filename = os.path.join(folder, f'image_{photo_count}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Image saved：{filename}")
            photo_count += 1
            start_time = time.time()

            # After saving the image, display a white image
            white_frame = np.ones_like(frame) * 255
            cv2.imshow('Calibration', white_frame)
            cv2.waitKey(100)
            cv2.imshow('Calibration', frame)

    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera(folder):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
    objp = np.zeros((num_vertical * num_horizontal, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_horizontal, 0:num_vertical].T.reshape(-1, 2) * grid_width
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # Read images
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    for fname in images:
        img = cv2.imread(fname)
        print("Image: ", fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (num_horizontal, num_vertical), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print('conership:', corners.shape[0])
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (num_horizontal, num_vertical), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    if not objpoints or not imgpoints:
        print("Not enough corners were detected for calibration")
        time.sleep(2)
        sys.exit(1)

    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print(f"Reprojection error: {mean_error}")

    # Check if calibration is qualified
    if mean_error < 1:
        np.savez("camera_params.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print("Camera calibrated")
    else:
        print("Calibration failed, please try again")

    return ret, mtx, dist, rvecs, tvecs


# Ask the user to input the camera ID
camera_id = int(input("Please enter your camera ID: "))
num_images = int(input("Please enter the number of photos you want to take: "))

capture_calibration_images(camera_id, num_images)

ret, mtx, dist, rvecs, tvecs = calibrate_camera('cali_imgs')
if ret:
    print("Calibration successful! Camera parameters are saved in 'camera_params.npz' file.")
    np.set_printoptions(precision=4, suppress=True)
    print('Camera Intrinsics Matrix:\n[[fx\t0\tcx]\n[0\tfy\tcy]\n[0\t0\t1]]  =\n', np.round(mtx, 4))
    print('dist:', np.round(dist, 4))