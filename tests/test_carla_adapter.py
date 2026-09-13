"""
Ground-truth tests for CarlAnomaly adapter.
"""
import numpy as np
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import CameraAgent, GNSSAgent

def test_carla_adapter():
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=5)
    
    # Test dataset generation
    normal_frames = dataset.generate_frames("NORMAL")
    assert len(normal_frames) == 5
    assert normal_frames[0].anomaly_type == "NORMAL"
    assert normal_frames[0].rgb.shape == (1, 2)
    
    # Test agent contracts
    cam_agent = CameraAgent()
    pred = cam_agent.predict(normal_frames[0].rgb)
    assert hasattr(pred, "value")
    
    unc_result = cam_agent.estimate_uncertainty(normal_frames[0].rgb)
    assert isinstance(unc_result.epistemic, float)
    assert isinstance(unc_result.aleatoric, float)
    
    # Test specific anomaly behavior
    blackout_frames = dataset.generate_frames("CAMERA_BLACKOUT")
    blackout_unc = cam_agent.estimate_uncertainty(blackout_frames[0].rgb)
    blackout_ep = blackout_unc.epistemic
    
    gnss_agent = GNSSAgent()
    gnss_unc = gnss_agent.estimate_uncertainty(blackout_frames[0].gnss)
    gnss_ep = gnss_unc.epistemic
    
    # Camera epistemic should be much higher than GNSS epistemic on a camera blackout
    assert blackout_ep > gnss_ep * 5, f"Camera EP ({blackout_ep}) not > GNSS EP ({gnss_ep})"
    print("[PASS] Carla Adapter Ground Truth")

if __name__ == "__main__":
    test_carla_adapter()
