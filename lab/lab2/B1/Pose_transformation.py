import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    # Define rotation angle and axis
    theta = np.pi/4  # 45 degrees
    axis = np.array([1, 1, 1]) / np.sqrt(3)  # unit vector along diagonal axis

    # Construct rotation matrix
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = axis
    R = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                  [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                  [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])

    # Plot transformed basis vectors
    x_axis = np.dot(R, np.array([1, 0, 0]))
    y_axis = np.dot(R, np.array([0, 1, 0]))
    z_axis = np.dot(R, np.array([0, 0, 1]))

    # Define set of points to rotate
    n_points = 1
    points = np.random.normal(size=(n_points, 3))

    # Apply rotation matrix to points
    rotated_points = np.dot(R, points.T).T

    # Visualize original and rotated points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='red')
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='green')
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='blue')
    ax.scatter(points[:,0], points[:,1], points[:,2], color='blue')
    ax.scatter(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], color='red')

    # Plot axis of rotation
    ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], color='black')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()