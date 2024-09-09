import cv2
import numpy as np

# Load the image
img = cv2.imread('image.jpg')

# # Noise Reduction
# # Apply Gaussian filter
# filtered = cv2.GaussianBlur(img, (5, 5), 0)
#
# # Display the original and filtered images side by side
# cv2.imshow('Original', img)
# cv2.imshow('Filtered', filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Correction
# # Apply histogram equalization to each color channel
# equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# equalized[:, :, 0] = cv2.equalizeHist(equalized[:, :, 0])
# equalized = cv2.cvtColor(equalized, cv2.COLOR_YUV2BGR)
#
# # Display the original and equalized images side by side
# cv2.imshow('Original', img)
# cv2.imshow('Equalized', equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Enhancement
# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply a histogram equalization to enhance the image contrast
# enhanced = cv2.equalizeHist(gray)
#
# # Show the original and enhanced images
# cv2.imshow('Original Image', gray)
# cv2.imshow('Enhanced Image', enhanced)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Segmentation
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the image
_, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Show the original and thresholded images
cv2.imshow('Original Image', img)
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()