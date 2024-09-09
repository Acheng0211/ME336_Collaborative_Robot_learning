from matplotlib.patches import Ellipse
from itertools import cycle
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr.utils import check_random_state
import cv2
import time
import numpy as np
import math
import cv2.aruco as aruco
from svgpathtools import svg2paths  # pip install svgpathtools
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM


def action_data_collection():
    aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters_create()
    '''
    If an error occurs, please use the following two lines of code.
    如果报错，请将上面两行代码注释掉，用后面这两行代码
    '''
    # aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # aruco_params = aruco.DetectorParameters()

    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    # Initialize camera capture
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(cap.get(3), cap.get(4), cap.get(5))

    d23_min = 1
    d45_min = 1
    d23_max = 0
    d45_max = 0
    calibFlag = False
    # Define an empty list to store the time series data
    marker_poses = []

    # Get the start time
    start_time = time.time()

    # Load the result from camera calibration
    data = np.load('/home/hgj/ME336/lab/lab1/image/camera_params.npz')
    mtx = data['mtx']
    dist = data['dist']

    force_left_finger = 0
    force_right_finger = 0
    midpointTvec = np.array([0, 0, 0])
    trai_datas = []
    trai_data = np.array([0, 0])  # x,y for the midPoint for marker 0,1
    force_datas = []
    force_data = np.array([0, 0])

    # Play the video
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        if ret == True:
            # Get the current time
            current_time = time.time()
            # Detect ArUco markers in the frame
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            if ids is not None:
                # Loop over all detected markers
                rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
                for i in range(0, ids.size):
                    # draw axis for the aruco markers
                    cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)
                if 2 in ids and 3 in ids:
                    marker2_idx = np.where(ids == 2)[0][0]
                    marker3_idx = np.where(ids == 3)[0][0]

                    t2 = tvecs[marker2_idx].reshape(-1)
                    t3 = tvecs[marker3_idx].reshape(-1)
                    d23 = math.sqrt((t2[0] - t3[0]) ** 2 + (t2[1] - t3[1]) ** 2)
                    if d23 < d23_min:
                        d23_min = d23
                        force_left_finger = 0
                    elif d23 > d23_max:
                        d23_max = d23
                        force_left_finger = 1
                    else:
                        force_left_finger = (d23 - d23_min) / (d23_max - d23_min)

                if 4 in ids and 5 in ids:
                    marker4_idx = np.where(ids == 4)[0][0]
                    marker5_idx = np.where(ids == 5)[0][0]

                    t4 = tvecs[marker4_idx].reshape(-1)
                    t5 = tvecs[marker5_idx].reshape(-1)
                    d45 = math.sqrt((t4[0] - t5[0]) ** 2 + (t4[1] - t5[1]) ** 2)
                    if d45 < d45_min:
                        d45_min = d45
                        force_right_finger = 0
                    elif d45 > d45_max:
                        d45_max = d45
                        force_right_finger = 1
                    else:
                        force_right_finger = (d45 - d45_min) / (d45_max - d45_min)

                if 3 in ids and 5 in ids:
                    marker3_idx = np.where(ids == 3)[0][0]
                    marker5_idx = np.where(ids == 5)[0][0]
                    midpointTvec = ((tvecs[marker3_idx] + tvecs[marker5_idx]) / 2)[0]
                    # print(midpointTvec)

                if 2 in ids and 3 in ids and 4 in ids and 5 in ids:
                    print("Force:", force_left_finger, force_right_finger)
                    if calibFlag:
                        traj_data = np.array([midpointTvec[0], midpointTvec[1]])
                        trai_datas.append(traj_data)
                        force_data = np.array([force_left_finger, force_right_finger])
                        force_datas.append(force_data)

                # Display the frame
                cv2.imshow('Frame', frame)
                # Exit on key press
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    calibFlag = True
                    print("Start Collection...")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Wait for the specified delay (in milliseconds)
                cv2.waitKey(1)
            else:
                break
    print(len(trai_datas))
    print(len(force_datas))

    np.save('trajdataForLFD_acheng.npy', trai_datas)
    np.save('forcedataForLFD_acheng.npy', force_datas)

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


def learning_from_demonstration():
    def make_demonstrations(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
                            start=np.zeros(2), goal=np.ones(2), random_state=None):
        """Generates demonstration that can be used to test imitation learning.
        Parameters
        ----------
        n_demonstrations : int
            Number of noisy demonstrations
        n_steps : int
            Number of time steps
        sigma : float, optional (default: 0.25)
            Standard deviation of noisy component
        mu : float, optional (default: 0.5)
            Mean of noisy component
        start : array, shape (2,), optional (default: 0s)
            Initial position
        goal : array, shape (2,), optional (default: 1s)
            Final position
        random_state : int
            Seed for random number generator
        Returns
        -------
        X : array, shape (n_task_dims, n_steps, n_demonstrations)
            Noisy demonstrated trajectories
        ground_truth : array, shape (n_task_dims, n_steps)
            Original trajectory
        """
        random_state = np.random.RandomState(random_state)

        X = np.empty((2, n_steps, n_demonstrations))

        # Generate ground-truth for plotting
        ground_truth = np.empty((2, n_steps))
        T = np.linspace(-0, 1, n_steps)
        ground_truth[0] = T
        ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                           np.exp(-0.5 * ((T - mu) / sigma) ** 2))

        # Generate trajectories
        for i in range(n_demonstrations):
            noisy_sigma = sigma * random_state.normal(1.0, 0.1)
            noisy_mu = mu * random_state.normal(1.0, 0.1)
            X[0, :, i] = T
            X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                              np.exp(-0.5 * ((T - noisy_mu) /
                                             noisy_sigma) ** 2))

        # Spatial alignment
        current_start = ground_truth[:, 0]
        current_goal = ground_truth[:, -1]
        current_amplitude = current_goal - current_start
        amplitude = goal - start
        ground_truth = ((ground_truth.T - current_start) * amplitude /
                        current_amplitude + start).T

        for demo_idx in range(n_demonstrations):
            current_start = X[:, 0, demo_idx]
            current_goal = X[:, -1, demo_idx]
            current_amplitude = current_goal - current_start
            X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
                                 amplitude / current_amplitude + start).T

        return X, ground_truth

    plot_covariances = True
    X, _ = make_demonstrations(
        n_demonstrations=10, n_steps=50, goal=np.array([1., 2.]),
        random_state=0)
    X = X.transpose(2, 1, 0)
    steps = X[:, :, 0].mean(axis=0)
    expected_mean = X[:, :, 1].mean(axis=0)
    expected_std = X[:, :, 1].std(axis=0)

    n_demonstrations, n_steps, n_task_dims = X.shape
    X_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
    X_train[:, :, 1:] = X
    t = np.linspace(0, 1, n_steps)
    X_train[:, :, 0] = t
    X_train = X_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)

    random_state = check_random_state(0)
    n_components = 4
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    bgmm = BayesianGaussianMixture(n_components=n_components, max_iter=100).fit(X_train)
    gmm = GMM(
        n_components=n_components,
        priors=bgmm.weights_,
        means=bgmm.means_,
        covariances=bgmm.covariances_,
        random_state=random_state)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Confidence Interval from GMM")

    plt.plot(X[:, :, 0].T, X[:, :, 1].T, c="k", alpha=0.1)

    means_over_time = []
    y_stds = []
    for step in t:
        conditional_gmm = gmm.condition([0], np.array([step]))
        conditional_mvn = conditional_gmm.to_mvn()
        means_over_time.append(conditional_mvn.mean)
        y_stds.append(np.sqrt(conditional_mvn.covariance[1, 1]))
        samples = conditional_gmm.sample(100)
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
    means_over_time = np.array(means_over_time)
    y_stds = np.array(y_stds)

    plt.plot(means_over_time[:, 0], means_over_time[:, 1], c="r", lw=2)
    plt.fill_between(
        means_over_time[:, 0],
        means_over_time[:, 1] - 1.96 * y_stds,
        means_over_time[:, 1] + 1.96 * y_stds,
        color="r", alpha=0.5)

    if plot_covariances:
        colors = cycle(["r", "g", "b"])
        for factor in np.linspace(0.5, 4.0, 8):
            new_gmm = GMM(
                n_components=len(gmm.means), priors=gmm.priors,
                means=gmm.means[:, 1:], covariances=gmm.covariances[:, 1:, 1:],
                random_state=gmm.random_state)
            for mean, (angle, width, height) in new_gmm.to_ellipses(factor):
                ell = Ellipse(xy=mean, width=width, height=height,
                              angle=np.degrees(angle))
                ell.set_alpha(0.15)
                ell.set_color(next(colors))
                plt.gca().add_artist(ell)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.subplot(122)
    plt.title("Confidence Interval from Raw Data")
    plt.plot(X[:, :, 0].T, X[:, :, 1].T, c="k", alpha=0.1)

    plt.plot(steps, expected_mean, c="r", lw=2)
    plt.fill_between(
        steps,
        expected_mean - 1.96 * expected_std,
        expected_mean + 1.96 * expected_std,
        color="r", alpha=0.5)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.show()


def plot():
    """
    =============================================
    Generate Time-Invariant Trajectories with GMR
    =============================================

    Make sure to run this example from gmr's root directory. An additional package
    is required to load an SVG file: 'svgpathtools'. We will further use the
    Bayesian GMM from sklearn to get a better fit of the data.

    We will load an SVG that contains one path that will be used to generate our
    training data. We extract a sequence of 2D points from the path. We will then
    compute the differences between points and assume they are velocities between
    those points. We fit a GMM on samples that contain four features: x- and
    y-coordinate of the position and the corresponding velocity. We now have a
    time-invariant representation of a trajectory. Starting from some position
    (x, y) we can compute a conditional GMM over the velocities. We can sample
    from the conditional GMM to generate a velocity, integrate the velocity
    to obtain a new position, and repeat this procedure as long as we want.

    Note that we use a "safe" sampling procedure here: we neglect Gaussians that
    have low prior probability and we resample from a selected Gaussian until
    we have a velocity that lies within the 70 % confidence interval of the
    Gaussian. We do this to avoid divergence from the training data.
    """
    print(__doc__)
    dt = 1.0
    forcedata = np.load("forcedataForLFD.npy")
    trajdata = np.load("trajdataForLFD.npy")
    # X = forcedata
    X = trajdata
    X_dot = (X[2:] - X[:-2]) / dt
    X = X[1:-1]
    X_train = np.hstack((X, X_dot))

    random_state = np.random.RandomState(0)
    n_components = 15

    bgmm = BayesianGaussianMixture(
        n_components=n_components, max_iter=500,
        random_state=random_state).fit(X_train)
    gmm = GMM(n_components=n_components, priors=bgmm.weights_, means=bgmm.means_,
              covariances=bgmm.covariances_, random_state=random_state)

    sampled_path = []
    x = X[0]
    # x = np.array([75.0, 90.0])  # left bottom
    sampling_dt = 0.2  # increases sampling frequency
    for t in range(500):
        sampled_path.append(x)
        cgmm = gmm.condition([0, 1], x)
        # default alpha defines the confidence region (e.g., 0.7 -> 70 %)
        x_dot = cgmm.sample_confidence_region(1, alpha=0.7)[0]
        x = x + sampling_dt * x_dot
    sampled_path = np.array(sampled_path)

    plt.plot(X[:, 0], X[:, 1], alpha=0.2, label="Demonstration")
    plt.plot(sampled_path[:, 0], sampled_path[:, 1], label="Reproduction")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #action_data_collection()
    # learning_from_demonstration()
    plot()
