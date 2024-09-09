import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Generate sample data
    data = np.random.rand(4, 6)

    def plot_motion_6d(data, draw_rpy=True):
        # Split data into x, y, z, roll, pitch, and yaw
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        roll = data[:, 3]
        pitch = data[:, 4]
        yaw = data[:, 5]

        # Plot data in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)

        # Add orientation information
        if draw_rpy:
            for i in range(len(data)):
                R = np.array([[np.cos(yaw[i])*np.cos(pitch[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.sin(roll[i]) - np.sin(yaw[i])*np.cos(roll[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.cos(roll[i]) + np.sin(yaw[i])*np.sin(roll[i])],
                              [np.sin(yaw[i])*np.cos(pitch[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.sin(roll[i]) + np.cos(yaw[i])*np.cos(roll[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.cos(roll[i]) - np.cos(yaw[i])*np.sin(roll[i])],
                              [-np.sin(pitch[i]), np.cos(pitch[i])*np.sin(roll[i]), np.cos(pitch[i])*np.cos(roll[i])]])
                x_end = x[i] + R[0, 0]
                y_end = y[i] + R[1, 0]
                z_end = z[i] + R[2, 0]
                ax.plot([x[i], x_end], [y[i], y_end], [z[i], z_end], 'r')
                x_end = x[i] + R[0, 1]
                y_end = y[i] + R[1, 1]
                z_end = z[i] + R[2, 1]
                ax.plot([x[i], x_end], [y[i], y_end], [z[i], z_end], 'g')
                x_end = x[i] + R[0, 2]
                y_end = y[i] + R[1, 2]
                z_end = z[i] + R[2, 2]
                ax.plot([x[i], x_end], [y[i], y_end], [z[i], z_end], 'b')

        plt.show()

    plot_motion_6d(data,draw_rpy=True)