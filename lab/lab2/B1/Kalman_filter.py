import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    # Generate noisy data from the function y=x^2
    x = np.linspace(0, 10, 100)
    noise = np.random.normal(loc=0, scale=10, size=100)
    y = x**2 + noise

    # Initialize the Kalman filter
    x_hat = np.zeros(2)
    P = np.diag([100, 100])
    F = np.array([[1, 1], [0, 1]])
    Q = np.diag([1, 1])
    H = np.array([1, 0]).reshape(1, 2)
    R = np.array([100]).reshape(1, 1)

    # Apply the Kalman filter to the noisy data
    filtered_y = []
    for i in range(len(x)):
        # Predict the next state
        x_hat = F.dot(x_hat)
        P = F.dot(P).dot(F.T) + Q

        # Update the state estimate based on the measurement
        y_hat = H.dot(x_hat)
        K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))
        x_hat = x_hat + K.dot(y[i] - y_hat)
        P = (np.eye(2) - K.dot(H)).dot(P)

        filtered_y.append(H.dot(x_hat))

    # Connect the noisy data points in order
    noisy_data = np.column_stack((x, y))
    sorted_noisy_data = noisy_data[noisy_data[:,0].argsort()]

    # Plot the original noisy data and the filtered data
    plt.plot(sorted_noisy_data[:,0], sorted_noisy_data[:,1], 'b-', label='Noisy data')
    plt.plot(x, filtered_y, 'r-', label='Filtered data')
    plt.legend()
    plt.show()