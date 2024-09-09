import cv2
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the filename of the npz file
filename = 'marker_poses_tactile66.npz'

# Load the npz file with allow_pickle=True
data = np.load(filename, allow_pickle=True)

# Extract the marker poses time series data
marker_poses = data['marker_poses']

# Init pose
relativePoses = []
rvecsForMarker4 = []
rvecsForMarker5 = []
tvecsForMarker4 = []
tvecsForMarker5 = []
force = []

#########Code below is for you to see the structure for the data
# # Print the ID and pose data for each marker at each time point
# for i, marker_data in enumerate(marker_poses):
#     print(f'Time {marker_data["time"]} seconds:')
#     # Access the single integer value of the marker ID for this time point
#     marker_id = marker_data['id']
#     print(f'Marker {marker_id}:')
#     print(f'    rvec: {marker_data["rvec"][0][0]}')
#     print(f'    tvec: {marker_data["tvec"][0][0]}')
#     print(f'    forceSense: {marker_data["tactile"]}')
#########

for i, marker_data in enumerate(marker_poses):
    # calculate relative pose between markers
    marker_id = marker_data['id']
    if marker_id == 4:
        rvecsForMarker4 = marker_data["rvec"][0][0]
        tvecsForMarker4 = marker_data["tvec"][0][0]
    elif marker_id == 5:
        rvecsForMarker5 = marker_data["rvec"][0][0]
        tvecsForMarker5 = marker_data["tvec"][0][0]

        R1, _ = cv2.Rodrigues(rvecsForMarker4)
        R2, _ = cv2.Rodrigues(rvecsForMarker5)
        t1 = tvecsForMarker4.reshape(-1)
        t2 = tvecsForMarker5.reshape(-1)
        R_rel = np.dot(R2.T, R1)
        t_rel = np.dot(-R2.T, t1) + np.dot(R2.T, t2)

        # convert relative rotation matrix to rotation vector
        rvec_rel, _ = cv2.Rodrigues(R_rel)
        rvec_rel = np.array([rvec_rel[0][0],rvec_rel[1][0],rvec_rel[2][0]])

        # format relative pose as 6-dimensional array
        relativePose = np.concatenate((rvec_rel, t_rel)).reshape(1, 6)[0]

        relativePoses.append(relativePose)
        force.append(marker_data["tactile"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(relativePoses, force, test_size=0.2, random_state=42)

# Create the MLPRegressor model with 2 hidden layers, each with 50 neurons
mlp = MLPRegressor(hidden_layer_sizes=(50,30), max_iter=5000, random_state=42, learning_rate_init=0.0051)

# Train the model on the training set
mlp.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = mlp.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

