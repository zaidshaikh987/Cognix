"""
CarlAnomalyDataset — synthetic replica mode.
Generates statistically faithful frames mimicking the CarlAnomaly dataset structure
so the exact same experimental pipeline runs without needing the 136GB download.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class CarlAnomalyFrame:
    rgb: np.ndarray
    depth: np.ndarray
    lidar: np.ndarray
    gnss: np.ndarray
    imu: np.ndarray
    segmentation: np.ndarray
    label: int  # 0 for safe, 1 for hazard
    anomaly_type: str
    timestep: int

class CarlAnomalyDataset:
    def __init__(self, mode="synthetic", n_frames_per_anomaly=200, seed=42):
        self.mode = mode
        self.n_frames = n_frames_per_anomaly
        self.rng = np.random.default_rng(seed)
        self.anomalies = [
            "NORMAL",
            "HIGH_NOISE",
            "OOD",
            "MISSING_AGENT",
            "PARTIAL_FAILURE",
            "MULTIPLE_FAILURE",
            "CONFLICT"
        ]
        
    def generate_frames(self, anomaly_type: str) -> list[CarlAnomalyFrame]:
        frames = []
        for t in range(self.n_frames):
            # Base healthy features (Safe = label 0)
            rgb_f = np.array([0.5, 0.5]) + self.rng.normal(0, 0.08, 2)
            depth_f = np.array([0.8, 0.0]) + self.rng.normal(0, 0.07, 2)
            lidar_f = np.array([0.5, 0.0]) + self.rng.normal(0, 0.08, 2)
            gnss_f = np.array([0.05, 0.0]) + self.rng.normal(0, 0.06, 2)
            imu_f = np.array([0.05, 0.0]) + self.rng.normal(0, 0.05, 2)
            seg_f = np.array([0.3, 0.0]) + self.rng.normal(0, 0.09, 2)
            
            label = 0
            
            if anomaly_type == "HIGH_NOISE":
                rgb_f += self.rng.normal(0, 0.2, 2)
                depth_f += self.rng.normal(0, 0.2, 2)
                lidar_f += self.rng.normal(0, 0.2, 2)
                gnss_f += self.rng.normal(0, 0.2, 2)
                imu_f += self.rng.normal(0, 0.2, 2)
                seg_f += self.rng.normal(0, 0.2, 2)
            
            elif anomaly_type == "OOD":
                rgb_f += np.array([1.5, -1.5])
                depth_f += np.array([-1.0, 1.0])
                label = 1
                
            elif anomaly_type == "MISSING_AGENT":
                # Hard dropout of camera
                rgb_f = np.zeros(2)
                label = 1
                
            elif anomaly_type == "PARTIAL_FAILURE":
                # Camera blackout (high variance)
                rgb_f = np.array([0.1, 0.1]) + self.rng.normal(0, 0.6, 2)
                label = 1
                
            elif anomaly_type == "MULTIPLE_FAILURE":
                # Camera + GNSS fail
                rgb_f = np.array([0.1, 0.1]) + self.rng.normal(0, 0.6, 2)
                gnss_f = np.array([0.9, 0.0]) + self.rng.normal(0, 0.6, 2)
                label = 1
                
            elif anomaly_type == "CONFLICT":
                # Camera, LiDAR, Depth say Hazard (1), GNSS, IMU, Seg say Safe (0)
                # But actually here we just simulate conflicting features
                rgb_f = np.array([0.1, 0.9]) + self.rng.normal(0, 0.1, 2) # Hazard
                lidar_f = np.array([0.1, 0.9]) + self.rng.normal(0, 0.1, 2) # Hazard
                depth_f = np.array([0.1, 0.9]) + self.rng.normal(0, 0.1, 2) # Hazard
                gnss_f = np.array([0.9, 0.1]) + self.rng.normal(0, 0.1, 2) # Safe
                imu_f = np.array([0.9, 0.1]) + self.rng.normal(0, 0.1, 2) # Safe
                seg_f = np.array([0.9, 0.1]) + self.rng.normal(0, 0.1, 2) # Safe
                label = 1 # Ground truth is hazard
                
            frame = CarlAnomalyFrame(
                rgb=np.array([rgb_f]),
                depth=np.array([depth_f]),
                lidar=np.array([lidar_f]),
                gnss=np.array([gnss_f]),
                imu=np.array([imu_f]),
                segmentation=np.array([seg_f]),
                label=label,
                anomaly_type=anomaly_type,
                timestep=t
            )
            frames.append(frame)
        return frames

    def get_all_scenarios(self):
        scenarios = {}
        for a in self.anomalies:
            scenarios[a] = self.generate_frames(a)
        return scenarios
