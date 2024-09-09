import numpy as np
import cv2

if __name__ == '__main__':

    # Set the chessboard size and grid width
    chessboard_size = (11, 8)
    grid_width = 0.007

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (10,7,0)
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * grid_width

    # Create arrays to store object points and image points
    obj_points = [] # 3D points in real world space
    img_points = [] # 2D points in image plane

    # Initialize camera and window
    cap = cv2.VideoCapture(2)
    cv2.namedWindow('camera')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Capture frames from camera
    while True:
        ret, frame = cap.read()

        # Find corners in current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # Draw corners on frame and display it
        if ret == True:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.imshow('camera', frame)

        # Save image if 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if ret == True:
                obj_points.append(objp)
                img_points.append(corners)
                cv2.imwrite('calibration{}.jpg'.format(len(obj_points)), frame)
                print("Image saved")

        # Calibrate camera if enough images are saved
        if len(obj_points) >= 20:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

            # Calculate reprojection error
            mean_error = 0
            for i in range(len(obj_points)):
                imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            mean_error /= len(obj_points)
            print(f"Reprojection error: {mean_error}")

            # Check if calibration is qualified
            if mean_error < 1:
                np.savez("camera_params.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                print("Camera calibrated")
                break
            else:
                print("Calibration failed, please try again")

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()