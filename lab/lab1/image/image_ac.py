import cv2


if __name__ == '__main__':

    # Initialize camera
    cap = cv2.VideoCapture(2)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video capture")
        exit()

    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CV_CAP_PROP_FPS, 30);
    # cap.set(cv2.CV_CAP_PROP_BRIGHTNESS, 1);
    # cap.set(cv2.CV_CAP_PROP_CONTRAST,40);
    # cap.set(cv2.CV_CAP_PROP_SATURATION, 50);
    # cap.set(cv2.CV_CAP_PROP_HUE, 50);
    # cap.set(cv2.CV_CAP_PROP_EXPOSURE, 50);

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Camera', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Check if 's' key is pressed to save image
        if key == ord('s'):
            cv2.imwrite('image.jpg', frame)
            print("Image captured!")
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()