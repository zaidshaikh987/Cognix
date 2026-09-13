"""
CarlAnomaly Evaluation Script
Runs COGNIX on the CarlAnomaly dataset and produces the Epistemic Specificity Plot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from cognix.engine.decision_engine import DecisionEngine
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import (
    CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent
)
from cognix.config.schema import CognixConfig

def plot_specificity(results, output_dir):
    anomalies = list(results.keys())
    affected = [results[a]['affected_epi'] for a in anomalies]
    unaffected = [results[a]['unaffected_epi'] for a in anomalies]
    
    x = np.arange(len(anomalies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, affected, width, label='Affected Agents (Mean Epistemic)', color='firebrick')
    rects2 = ax.bar(x + width/2, unaffected, width, label='Unaffected Agents (Mean Epistemic)', color='steelblue')
    
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title('COGNIX Sensor-Specific Epistemic Response on CarlAnomaly')
    ax.set_xticks(x)
    ax.set_xticklabels(anomalies, rotation=45, ha='right')
    ax.legend()
    
    # Annotate ratios
    for i in range(len(anomalies)):
        ratio = affected[i] / (unaffected[i] + 1e-6)
        if ratio > 1.5:
            ax.text(i - width/2, affected[i] + 0.02, f"{ratio:.1f}x", ha='center', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epistemic_specificity.png'))
    plt.close()

def run_evaluation(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=20)
    scenarios = dataset.get_all_scenarios()
    
    agents = [
        CameraAgent(),
        DepthAgent(),
        LiDARAgent(),
        GNSSAgent(),
        IMUAgent(),
        SegAgent()
    ]
    
    config = CognixConfig()
    engine = DecisionEngine(config)
    
    # Mapping anomaly types to specifically affected agents
    anomaly_mapping = {
        "NORMAL": [],
        "CAMERA_BLACKOUT": ["Camera"],
        "GPS_DRIFT": ["GNSS"],
        "HEAVY_RAIN": ["LiDAR", "Camera"],
        "MULTI_FAILURE": ["Camera", "LiDAR", "GNSS"]
    }
    
    summary_results = {}
    
    for anomaly_type, frames in scenarios.items():
        affected_agents = anomaly_mapping[anomaly_type]
        all_affected_epis = []
        all_unaffected_epis = []
        decisions = []
        
        for frame in frames:
            inputs = {
                "Camera": frame.rgb,
                "Depth": frame.depth,
                "LiDAR": frame.lidar,
                "GNSS": frame.gnss,
                "IMU": frame.imu,
                "Seg": frame.segmentation
            }
            
            # Calculate epistemic for plotting
            frame_affected = []
            frame_unaffected = []
            for agent in agents:
                ep = agent.estimate_uncertainty(inputs).epistemic
                if agent.agent_id in affected_agents:
                    frame_affected.append(ep)
                else:
                    frame_unaffected.append(ep)
            
            if frame_affected:
                all_affected_epis.append(np.mean(frame_affected))
            if frame_unaffected:
                all_unaffected_epis.append(np.mean(frame_unaffected))

            # Run engine for collective decision
            reliabilities = {agent.agent_id: 1.0 for agent in agents}
            result = engine.decide(agents, inputs, agent_reliabilities=reliabilities)
            decisions.append(result.decision.name)
        
        # If normal, use all agents as unaffected to get baseline
        if not all_affected_epis:
            mean_affected = np.mean(all_unaffected_epis) 
            mean_unaffected = np.mean(all_unaffected_epis)
        else:
            mean_affected = np.mean(all_affected_epis)
            mean_unaffected = np.mean(all_unaffected_epis)
        
        summary_results[anomaly_type] = {
            "affected_epi": mean_affected,
            "unaffected_epi": mean_unaffected,
            "most_common_decision": max(set(decisions), key=decisions.count)
        }
        
    print("\n--- RESULTS TABLE ---")
    print(f"{'Anomaly Type':<18} | {'Aff. Epi':<10} | {'Unaff. Epi':<10} | {'Ratio':<8} | {'Decision'}")
    for a, res in summary_results.items():
        if a == "NORMAL":
            ratio = 1.0
        else:
            ratio = res['affected_epi'] / (res['unaffected_epi'] + 1e-6)
            
        print(f"{a:<18} | {res['affected_epi']:<10.4f} | {res['unaffected_epi']:<10.4f} | {ratio:<7.1f}x | {res['most_common_decision']}")
        
    plot_specificity(summary_results, output_dir)
    print(f"\nPlot saved to {output_dir}/epistemic_specificity.png")

if __name__ == "__main__":
    run_evaluation("results/carla_anomaly")
