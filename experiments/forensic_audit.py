import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognix import (
    DecisionEngine, CognixConfig, UncertaintyDecomposition, 
    EpistemicWeightedFusion, ConformalPredictor, EscalationEngine
)
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent

def run_audit():
    print("Starting COGNIX Forensic Audit on Seed 42...")
    
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=20, seed=42)
    all_frames = dataset.get_all_scenarios()
    
    MODELS = {
        "NoGraph": {"method": "none"},
        "StandardGAT": {"method": "gat"},
        "EpistemicGAT": {"method": "epistemic_gat"}
    }
    
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"AUDITING MODEL: {model_name}")
        print(f"{'='*60}")
        
        config = CognixConfig()
        config.graph.method = cfg["method"]
        
        engine = DecisionEngine(
            config=config,
            uncertainty=UncertaintyDecomposition(),
            belief=EpistemicWeightedFusion(),
            calibrator=ConformalPredictor(),
            escalation=EscalationEngine(),
            graph=None,
            mode="production"
        )
        
        if cfg["method"] == "gat":
            from cognix.graph.epistemic_gat import EpistemicGAT
            engine.pipeline.graph = EpistemicGAT(use_epistemic_prior=False)
        elif cfg["method"] == "epistemic_gat":
            from cognix.graph.epistemic_gat import EpistemicGAT
            engine.pipeline.graph = EpistemicGAT(use_epistemic_prior=True)
            
        # Mock calibration for conformal
        cal_outputs = np.random.uniform(0, 1, (100, 2))
        cal_labels = np.random.randint(0, 2, 100)
        engine.pipeline.calibrator.fit(cal_outputs=cal_outputs, cal_labels=cal_labels)
        
        print(f"\n--- Conformal Calibration Info ---")
        print(f"N_cal: {engine.pipeline.calibrator.n_cal}")
        
        # Train graph if present
        if hasattr(engine.pipeline.graph, "fit"):
            train_frames = all_frames["NORMAL"][:25] + all_frames["OOD"][:25]
            X_train = []
            y_train = []
            for f in train_frames:
                inputs = {
                    "Camera": f.rgb, "Depth": f.depth, "LiDAR": f.lidar,
                    "GNSS": f.gnss, "IMU": f.imu, "Seg": f.segmentation
                }
                X_train.append(inputs) 
                y_train.append(f.label)
            agents_list = [CameraAgent(), DepthAgent(), LiDARAgent(), GNSSAgent(), IMUAgent(), SegAgent()]
            engine.pipeline.graph.fit(agents_list, X_train, y_train, epochs=30, verbose=False)
            
        for scenario in ["NORMAL", "OOD"]:
            print(f"\n--- Scenario: {scenario} ---")
            frames = all_frames[scenario]
            
            set_sizes = {1: 0, 2: 0}
            probs = []
            
            for idx, frame in enumerate(frames):
                inputs = {
                    "Camera": frame.rgb, "Depth": frame.depth, "LiDAR": frame.lidar,
                    "GNSS": frame.gnss, "IMU": frame.imu, "Seg": frame.segmentation
                }
                agents = [CameraAgent(), DepthAgent(), LiDARAgent(), GNSSAgent(), IMUAgent(), SegAgent()]
                
                result = engine.decide(agents, inputs, {a.agent_id: 1.0 for a in agents})
                
                # Extract prob
                prob = result.confidence
                probs.append(prob)
                
                # Check real prediction set size
                if result.calibration_metrics and "prediction_set" in result.calibration_metrics:
                    ps = result.calibration_metrics["prediction_set"]
                    set_sizes[len(ps)] = set_sizes.get(len(ps), 0) + 1
                
                if idx < 3: # Print first 3 frames data flow
                    print(f"\n  [Frame {idx}] True Label: {frame.label}")
                    print(f"    Raw Agent Preds: {result.agent_predictions}")
                    if engine.pipeline.graph:
                        # Since we can't easily intercept the middle of the pipeline without modifying code,
                        # we print what we can from result.
                        print(f"    Refined Prob: {prob:.4f} | Total Unc: {result.total_uncertainty:.4f}")
                    if result.calibration_metrics:
                        print(f"    Prediction Set: {result.calibration_metrics['prediction_set']}")
            
            probs = np.array(probs)
            print(f"\n  Stats for {scenario}:")
            print(f"    Probabilities: min={np.min(probs):.4f}, max={np.max(probs):.4f}, mean={np.mean(probs):.4f}, std={np.std(probs):.4f}")
            print(f"    Conformal Set Sizes: {set_sizes}")

if __name__ == "__main__":
    run_audit()
