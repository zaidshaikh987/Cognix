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
            "CAMERA_BLACKOUT",
            "GPS_DRIFT",
            "HEAVY_RAIN",
            "MULTI_FAILURE"
        ]
        
    def generate_frames(self, anomaly_type: str) -> list[CarlAnomalyFrame]:
        frames = []
        for t in range(self.n_frames):
            # Base features: healthy
            # feature format: [metric1, metric2]
            rgb_f = np.array([0.5, 0.5]) + self.rng.normal(0, 0.02, 2)
            depth_f = np.array([0.8, 0.0]) + self.rng.normal(0, 0.02, 2)
            lidar_f = np.array([0.5, 0.0]) + self.rng.normal(0, 0.02, 2)
            gnss_f = np.array([0.05, 0.0]) + self.rng.normal(0, 0.01, 2)
            imu_f = np.array([0.05, 0.0]) + self.rng.normal(0, 0.01, 2)
            seg_f = np.array([0.3, 0.0]) + self.rng.normal(0, 0.02, 2)
            
            label = 0
            
            # Apply anomalies
            if anomaly_type == "CAMERA_BLACKOUT":
                # Camera variance goes to 0, brightness 0
                rgb_f = np.array([0.0, 0.0])
                label = 1
            elif anomaly_type == "GPS_DRIFT":
                # GNSS drift spikes
                gnss_f = np.array([0.9, 0.0]) + self.rng.normal(0, 0.05, 2)
                label = 1
            elif anomaly_type == "HEAVY_RAIN":
                # Lidar density drops/spikes, Camera gets blurred/glare
                lidar_f = np.array([0.1, 0.0]) + self.rng.normal(0, 0.05, 2)
                rgb_f = np.array([0.85, 0.85]) + self.rng.normal(0, 0.05, 2) 
                label = 1
            elif anomaly_type == "MULTI_FAILURE":
                rgb_f = np.array([0.0, 0.0])
                lidar_f = np.array([0.1, 0.0])
                gnss_f = np.array([0.9, 0.0])
                label = 1
                
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
