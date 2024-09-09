from sklearn.linear_model import LinearRegression
import numpy as np

# generate some random data with 6 input features and 6 output variables
X = np.random.rand(100, 6)
y = np.random.rand(100, 6)

# create a Linear Regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# predict outputs for new data using the trained model
X_new = np.array([[1, 2, 3, 4, 5, 6]])
y_pred = model.predict(X_new)

print("Predicted outputs for new data:")
print(y_pred)