"""
COGNIX Research Question 001: Epistemic Fusion Validation.

Validates the hypothesis that Epistemic-Weighted Fusion improves calibration
under out-of-distribution (OOD) shift, noise, and agent degradation, compared 
to baseline fusion methods.

This script uses configurable sample sizes, 5 random seeds, and actual
OOD generation rather than faking dropout probability.
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

from cognix import DecisionEngine
from cognix.config.schema import CognixConfig
from cognix.engine.provenance import MetricProvenance, ExperimentMetadata, DecisionTrace
from cognix.metrics.evaluation import calculate_ece, accuracy, brier_score, LatencyTracker

# ==========================================
# 1. DATA GENERATION
# ==========================================
class DataGenerator:
    """Generates synthetic multi-modal data for heterogeneous agents."""
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
    def generate_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """P_train(X): Standard normally distributed features."""
        # 3 distinct features
        X = self.rng.normal(0, 1, (self.n_samples, 3))
        # True state depends non-linearly on features
        logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2]**2
        probs = 1 / (1 + np.exp(-logits))
        y = (self.rng.random(self.n_samples) < probs).astype(float)
        return X, y
        
    def generate_test(self, condition: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P_test(X): Shifted based on condition."""
        X, y = self.generate_train()
        ood_labels = np.zeros(self.n_samples)
        
        if condition == "NORMAL":
            pass
        elif condition == "HIGH_NOISE":
            # Add severe Gaussian noise to Feature 1 (Agent B's main feature)
            X[:, 1] += self.rng.normal(0, 3, self.n_samples)
        elif condition == "OOD_SHIFT":
            # Covariate shift on Feature 1, moving it far outside training bounds [-3, 3]
            shift_idx = self.rng.choice(self.n_samples, size=int(0.5 * self.n_samples), replace=False)
            X[shift_idx, 1] += 10.0 # Extreme shift
            ood_labels[shift_idx] = 1.0
        elif condition == "MISSING_AGENT":
            # Feature 1 goes completely dark (zeros)
            X[:, 1] = 0.0
            
        return X, y, ood_labels

# ==========================================
# 2. HETEROGENEOUS MODELS
# ==========================================
class FeatureModel(nn.Module):
    def __init__(self, feature_idx: int):
        super().__init__()
        self.feature_idx = feature_idx
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Standard fixed dropout for UQ
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract only the specific feature this model is trained on
        feat = x[:, self.feature_idx:self.feature_idx+1]
        return self.net(feat)

class MultiFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class HeterogeneousAgent:
    """Agent wrapping a specific PyTorch architecture."""
    def __init__(self, name: str, model: nn.Module):
        self.agent_id = name
        self.name = name
        self.model = model
        self.passes = 30
        self.last_pred = 0.5
        self.last_epi = 0.0
        self.healthy = True
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).view(-1, 1)
        
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            
    def predict(self, x: np.ndarray) -> float:
        if not self.healthy:
            return 0.5
        self.model.eval()
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            self.last_pred = float(self.model(t_x).item())
        return self.last_pred
        
    def estimate_uncertainty(self, x: np.ndarray) -> Any:
        if not self.healthy:
            class UQ:
                epistemic = 1.0
                aleatoric = 1.0
                total = 2.0
            return UQ()
            
        self.model.train() # Enable Dropout
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            preds = [self.model(t_x).item() for _ in range(self.passes)]
            
        self.last_epi = float(np.var(preds))
        
        class UQ:
            epistemic = self.last_epi
            aleatoric = 0.05
            total = self.last_epi + 0.05
        return UQ()

# ==========================================
# 3. EVALUATION PROTOCOL
# ==========================================
def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    
    n_samples = args.samples
    seeds = [42, 101, 202, 303, 404]
    conditions = ["NORMAL", "HIGH_NOISE", "OOD_SHIFT", "MISSING_AGENT"]
    baselines = ["uniform", "confidence", "bayesian", "epistemic_weighted"]
    
    run_id = f"EXP-RQ0-{int(time.time())}"
    results_dir = os.path.join("results", "RQ_SYNTHETIC_001")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting RQ_SYNTHETIC_001 | Samples: {n_samples} | Seeds: {len(seeds)}")
    
    all_results = []
    dashboard_traces = [] # Will capture traces for the dashboard
    
    for seed in seeds:
        print(f"--- Running SEED {seed} ---")
        torch.manual_seed(seed)
        
        # 1. Generate Training Data & Train Heterogeneous Agents
        gen = DataGenerator(n_samples=n_samples, seed=seed)
        X_train, y_train = gen.generate_train()
        
        agents = [
            HeterogeneousAgent("Agent_A_Feat0", FeatureModel(0)),
            HeterogeneousAgent("Agent_B_Feat1", FeatureModel(1)), # Will face the OOD shift
            HeterogeneousAgent("Agent_C_Feat2", FeatureModel(2)),
            HeterogeneousAgent("Agent_D_AllFeat", MultiFeatureModel()),
        ]
        
        for agent in agents:
            agent.fit(X_train, y_train, epochs=100)
            
        # 2. Test across Conditions
        for cond in conditions:
            X_test, y_test, ood_labels = gen.generate_test(condition=cond)
            
            # Handle Missing Agent gracefully
            if cond == "MISSING_AGENT":
                agents[1].healthy = False
            else:
                agents[1].healthy = True
                
            # 3. Test across Baselines
            for bline in baselines:
                config = CognixConfig()
                engine = DecisionEngine(
                    config=config,
                    belief=bline,
                    attribution="epistemic_shapley"
                )
                
                tracker = LatencyTracker()
                preds = []
                
                for i in range(n_samples):
                    x_i = X_test[i]
                    # Engine decides
                    t0 = time.perf_counter()
                    res = engine.decide(agents, x_i)
                    tracker.record("total", (time.perf_counter() - t0) * 1000)
                    preds.append(res.confidence)
                    
                    # Capture traces for the dashboard (only for Seed 42, EWF baseline, to keep file size reasonable)
                    if seed == 42 and bline == "epistemic_weighted":
                        trace = DecisionTrace(
                            timestamp=time.time(),
                            input_id=f"{cond}_sample_{i}",
                            agent_predictions=res.agent_predictions,
                            uncertainty={
                                "epistemic": res.epistemic_uncertainty,
                                "aleatoric": res.aleatoric_uncertainty,
                                "total": res.total_uncertainty
                            },
                            belief=[1 - res.confidence, res.confidence],
                            weights=res.agent_trust_weights,
                            communication={},
                            calibration={"calibrated_confidence": res.calibrated_confidence},
                            risk=res.risk_level.value,
                            decision=res.decision.value,
                            attribution=res.agent_contributions,
                            latency=res.latency_ms
                        )
                        dashboard_traces.append(trace.to_dict())
                
                # Compute metrics
                ece = calculate_ece(np.array(preds), y_test)
                acc = accuracy(np.array(preds), y_test)
                brier = brier_score(np.array(preds), y_test)
                lat = tracker.get_percentiles("total")
                
                all_results.append({
                    "seed": seed,
                    "condition": cond,
                    "baseline": bline,
                    "ece": ece,
                    "accuracy": acc,
                    "brier": brier,
                    "latency_p95": lat["p95"]
                })
                
    # ==========================================
    # 4. AGGREGATION & EXPORT
    # ==========================================
    # Aggregate across seeds
    aggregated = {}
    for res in all_results:
        key = f"{res['condition']}_{res['baseline']}"
        if key not in aggregated:
            aggregated[key] = {"ece": [], "acc": [], "brier": [], "lat": []}
        aggregated[key]["ece"].append(res["ece"])
        aggregated[key]["acc"].append(res["accuracy"])
        aggregated[key]["brier"].append(res["brier"])
        aggregated[key]["lat"].append(res["latency_p95"])
        
    # Build a sample MetricProvenance for the dashboard to render
    dash_metrics = {
        "cognix_ece": {"value": float(np.mean(aggregated.get("OOD_SHIFT_epistemic_weighted", {}).get("ece", [0.0])))},
        "baseline_ece": {"value": float(np.mean(aggregated.get("OOD_SHIFT_uniform", {}).get("ece", [0.0])))},
        "cognix_acc": float(np.mean(aggregated.get("OOD_SHIFT_epistemic_weighted", {}).get("acc", [0.0]))),
        "baseline_acc": float(np.mean(aggregated.get("OOD_SHIFT_uniform", {}).get("acc", [0.0])))
    }
    
    if dashboard_traces:
        dashboard_traces[-1]["metrics"] = dash_metrics
        with open(os.path.join(results_dir, "trace.json"), "w") as f:
            json.dump(dashboard_traces, f, indent=2)
            
        with open(os.path.join("results", "latest_run.txt"), "w") as f:
            f.write("RQ_SYNTHETIC_001")
        
    final_metrics = []
    for key, vals in aggregated.items():
        cond, bline = key.split("_", 1)
        final_metrics.append({
            "condition": cond,
            "baseline": bline,
            "ece_mean": float(np.mean(vals["ece"])),
            "ece_std": float(np.std(vals["ece"])),
            "acc_mean": float(np.mean(vals["acc"])),
            "acc_std": float(np.std(vals["acc"])),
            "brier_mean": float(np.mean(vals["brier"])),
            "brier_std": float(np.std(vals["brier"])),
            "latency_p95_mean": float(np.mean(vals["lat"])),
        })
        
    # Save to disk
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    meta = ExperimentMetadata(
        run_id=run_id,
        dataset="SYNTHETIC_SHIFT_3D",
        dataset_version="1.0",
        model="Heterogeneous_MCDropout_Ensemble",
        model_version="1.0",
        experiment="RQ_SYNTHETIC_001",
        seed=0, # Aggregated
        configuration={"n_samples": n_samples, "seeds": seeds, "conditions": conditions}
    )
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
        
    print(f"\nExecution Complete. Data saved to {results_dir}")
    print("Use `python examples/generate_plots.py` to view results.")
    
if __name__ == "__main__":
    run_evaluation()
