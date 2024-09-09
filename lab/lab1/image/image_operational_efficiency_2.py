import cv2
import time

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create the MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Load the video
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Initialize variables for time measurement
frame_count = 0
total_time = 0

# Read the first frame
ret, frame = cap.read()

# Detect the initial face(s) using the Haar Cascade classifier
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Initialize the tracker with the first face (if any)
if len(faces) > 0:
    # Select the first face
    (x, y, w, h) = faces[0]
    # Initialize the tracker with the first face
    tracker.init(frame, (x, y, w, h))

# Loop over each frame in the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # If we reached the end of the video, break out of the loop
    if not ret:
        break

    # Start the timer
    start_time = time.time()

    # Track the face using the MOSSE tracker (if initialized)
    if tracker:
        # Update the tracker with the current frame
        ok, bbox = tracker.update(frame)
        # If the tracking was successful, draw a rectangle around the face
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # If the tracking failed (e.g., the face went out of the frame), reset the tracker
        else:
            tracker = None

    # If the tracker is not initialized or failed, detect faces using the Haar Cascade classifier
    if not tracker:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # If a face is detected, initialize the tracker with the first face
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker = cv2.TrackerMOSSE_create()
            tracker.init(frame, (x, y, w, h))

    # Show the frame with the detected/tracked face
    cv2.imshow('frame', frame)

    # Stop the timer and update the time variables
    end_time = time.time()
    total_time += end_time - start_time
    frame_count += 1

    # Wait for a key press, and exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources (camera and window)
cap.release()
cv2.destroyAllWindows()

# Calculate the average time cost per frame
average_time = total_time / frame_count
print('Average time cost per frame: {:.2f} seconds'.format(average_time))