"""
Kalman Filter for ByteTrack
Adapted for 2D bounding box tracking
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    """
    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        
        Args:
            measurement: ndarray, bounding box coordinates (x, y, a, h) where (x,y) is the center
                         and (a, h) is the aspect ratio and height
        
        Returns:
            mean: ndarray, initial state mean
            covariance: ndarray, initial state covariance
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: ndarray, the predicted state mean
            covariance: ndarray, the predicted state covariance
        
        Returns:
            mean: ndarray, the predicted state mean
            covariance: ndarray, the predicted state covariance
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        
        Args:
            mean: ndarray, the state mean
            covariance: ndarray, the state covariance
        
        Returns:
            mean: ndarray, the projected mean
            covariance: ndarray, the projected covariance
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.
        
        Args:
            mean: ndarray, the predicted state mean
            covariance: ndarray, the predicted state covariance
            measurement: ndarray, the measurement
        
        Returns:
            mean: ndarray, the corrected state mean
            covariance: ndarray, the corrected state covariance
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Use Cholesky decomposition to solve for Kalman gain
        chol_factor, lower = cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = cho_solve((chol_factor, lower), 
                                np.dot(self._update_mat, covariance.T)).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            mean: ndarray, state mean
            covariance: ndarray, state covariance
            measurements: ndarray, measurements
            only_position: bool, if True, distance computation is done with respect to position only
        
        Returns:
            distances: ndarray, an array of length N, where N is the number of measurements
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = np.linalg.solve(cholesky_factor, d.T).T
        squared_maha = np.sum(z * z, axis=1)
        return squared_maha

