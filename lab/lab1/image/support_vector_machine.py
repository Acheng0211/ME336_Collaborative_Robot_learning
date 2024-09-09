import numpy as np
from sklearn import svm

# Generate example input and output data
X = np.random.rand(100, 6)
y = np.random.randint(0, 6, 100)

# Create an SVM classifier using a radial basis function (RBF) kernel
clf = svm.SVC(kernel='rbf')

# Train the model on the input data
clf.fit(X, y)

# Predict the output values for new input data
new_input = np.array([[1, 2, 3, 4, 5, 6]])
prediction = clf.predict(new_input)

print("Prediction:", prediction)