"""
CarlAnomaly Feature Extractors

These functions simulate or process raw sensor data into the features
expected by the COGNIX sensor agents. In synthetic mode, they take 
simulated sensor arrays and extract meaningful metrics that simulate
real neural network outputs (e.g. ResNet18 embeddings).
"""
import numpy as np

def rgb_to_hazard_prob(image_batch: np.ndarray) -> np.ndarray:
    """
    Simulates ResNet18 embedding -> logistic regression for binary hazard.
    Synthetic feature format: [variance, brightness]
    """
    B = image_batch.shape[0]
    hazard_probs = np.zeros(B)
    for i in range(B):
        var, bright = image_batch[i, 0], image_batch[i, 1]
        # Normal is around (0.5, 0.5). Blackout is (0,0), Glare is (0.8+, 0.8+)
        hazard = 1.0 - np.exp(-10 * ((var - 0.5)**2 + (bright - 0.5)**2))
        hazard_probs[i] = np.clip(hazard, 0.01, 0.99)
    return hazard_probs

def depth_to_obstacle_score(depth_batch: np.ndarray) -> np.ndarray:
    """Analyzes depth map for obstacle proximity."""
    B = depth_batch.shape[0]
    scores = np.zeros(B)
    for i in range(B):
        min_depth = depth_batch[i, 0]
        scores[i] = np.clip(1.0 - min_depth, 0.01, 0.99)
    return scores

def lidar_to_density_score(lidar_batch: np.ndarray) -> np.ndarray:
    """Point cloud density + nearest obstacle."""
    B = lidar_batch.shape[0]
    scores = np.zeros(B)
    for i in range(B):
        density = lidar_batch[i, 0]
        # density deviates from 0.5 in anomalies like rain
        scores[i] = np.clip(abs(density - 0.5) * 2, 0.01, 0.99)
    return scores

def gnss_to_drift_score(gnss_batch: np.ndarray) -> np.ndarray:
    """Kalman-filtered GPS consistency check."""
    B = gnss_batch.shape[0]
    scores = np.zeros(B)
    for i in range(B):
        drift = gnss_batch[i, 0]
        scores[i] = np.clip(drift, 0.01, 0.99)
    return scores

def imu_to_motion_score(imu_batch: np.ndarray) -> np.ndarray:
    """Jerk detection on accelerometer data."""
    B = imu_batch.shape[0]
    scores = np.zeros(B)
    for i in range(B):
        jerk = imu_batch[i, 0]
        scores[i] = np.clip(jerk, 0.01, 0.99)
    return scores

def seg_to_complexity_score(seg_batch: np.ndarray) -> np.ndarray:
    """Object count + class entropy from segmentation."""
    B = seg_batch.shape[0]
    scores = np.zeros(B)
    for i in range(B):
        entropy = seg_batch[i, 0]
        scores[i] = np.clip(entropy, 0.01, 0.99)
    return scores
