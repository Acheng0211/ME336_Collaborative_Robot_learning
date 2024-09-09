import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate random X and y datasets
np.random.seed(42)
X = np.random.rand(100, 6)
y = np.random.rand(100, 6)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the MLPRegressor model with 2 hidden layers, each with 50 neurons
mlp = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)

# Train the model on the training set
mlp.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = mlp.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)