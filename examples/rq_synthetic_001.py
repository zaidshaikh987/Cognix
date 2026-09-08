"""
COGNIX Research Question 001: Epistemic Fusion Validation — Revision 3.

Validates the hypothesis that Epistemic-Weighted Fusion improves decision calibration
under out-of-distribution (OOD) shift, noise, and agent degradation, compared
to baseline fusion methods.

Changes in Revision 3:
- 3-way train/calibration/test split (60/20/20)
- Aleatoric uncertainty: Bernoulli predictive-variance decomposition (not 0.05)
- Per-agent reliability: validation accuracy on calibration split
- Conformal calibrator pre-fitted on collective post-GNN-fusion outputs
- Pipeline mode="research": no fallbacks, all modules must execute
- EpistemicGAT integrated via graph parameter
- Collective Shapley value function

Canonical pipeline: M1 -> M4 -> M2 -> M3 -> M5
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from cognix.engine.pipeline import CognixPipeline
from cognix.config.schema import CognixConfig
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.belief.fusion import EpistemicWeightedFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.metrics.evaluation import calculate_ece, accuracy, brier_score, LatencyTracker
from cognix.calibration.conformal import evaluate_coverage

# ==========================================
# 1. DATA GENERATION
# ==========================================
class DataGenerator:
    """Generates synthetic multi-modal data for heterogeneous agents."""

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """P_train(X): Standard normally distributed features."""
        X = self.rng.normal(0, 1, (self.n_samples, 3))
        logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] ** 2
        probs = 1 / (1 + np.exp(-logits))
        y = (self.rng.random(self.n_samples) < probs).astype(float)
        return X, y

    def apply_condition(
        self, X: np.ndarray, y: np.ndarray, condition: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply OOD condition to feature matrix."""
        X = X.copy()
        ood_labels = np.zeros(len(X))

        if condition == "NORMAL":
            pass
        elif condition == "HIGH_NOISE":
            X[:, 1] += self.rng.normal(0, 3, len(X))
        elif condition == "OOD_SHIFT":
            shift_idx = self.rng.choice(len(X), size=int(0.5 * len(X)), replace=False)
            X[shift_idx, 1] += 10.0
            ood_labels[shift_idx] = 1.0
        elif condition == "MISSING_AGENT":
            X[:, 1] = 0.0

        return X, y, ood_labels


# ==========================================
# 2. HETEROGENEOUS MODELS (MC Dropout)
# ==========================================
class FeatureModel(nn.Module):
    def __init__(self, feature_idx: int):
        super().__init__()
        self.feature_idx = feature_idx
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = x[:, self.feature_idx:self.feature_idx + 1]
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


class HeterogeneousAgent(AgentInterface):
    """
    Agent wrapping a specific PyTorch architecture with MC Dropout UQ.

    Uncertainty decomposition (Bernoulli predictive-variance decomposition):
        p_bar = mean(p_t)  for t = 1..T
        U_epistemic = mean((p_t - p_bar)^2)  [variance of MC samples]
        U_aleatoric = mean(p_t * (1 - p_t))  [expected Bernoulli variance]
        U_total     = U_epistemic + U_aleatoric

    This decomposition is motivated by Kendall & Gal (2017) and is explicitly
    defined here as the operational uncertainty measure for this experiment.
    """

    def __init__(self, name: str, model: nn.Module, T: int = 30):
        self.agent_id = name
        self.name = name
        self.model = model
        self.T = T  # MC Dropout passes
        self.last_pred = 0.5
        self.last_epi = 0.0
        self.last_ale = 0.0
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

    def predict(self, x: np.ndarray) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(value=0.5, confidence=0.0)
        self.model.eval()
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            self.last_pred = float(self.model(t_x).item())
        return PredictionResult(value=self.last_pred, confidence=1.0)

    def estimate_uncertainty(self, x: Any) -> UncertaintyResult:
        """
        Run T=30 MC Dropout forward passes and compute Bernoulli
        predictive-variance decomposition.
        """
        if not self.healthy:
            return UncertaintyResult(prediction=0.5, epistemic=1.0, aleatoric=1.0, total=2.0)

        # Must have a valid input
        if x is None:
            raise RuntimeError(
                f"Agent {self.agent_id}: estimate_uncertainty requires input_data "
                "in research mode. Received None."
            )

        self.model.train()  # Enable Dropout
        with torch.no_grad():
            t_x = torch.FloatTensor(x).view(1, -1)
            preds = np.array([self.model(t_x).item() for _ in range(self.T)])

        p_bar = np.mean(preds)
        epi = float(np.mean((preds - p_bar) ** 2))        # Var(p_t)
        ale = float(np.mean(preds * (1.0 - preds)))        # E[p_t(1-p_t)]
        total = epi + ale

        self.last_epi = epi
        self.last_ale = ale

        return UncertaintyResult(prediction=p_bar, epistemic=epi, aleatoric=ale, total=total)
        
    def metadata(self) -> dict:
        return {"id": self.agent_id}


# ==========================================
# 3. PRE-CALIBRATION PIPELINE RUNNER
# ==========================================
def run_pre_calibration(
    agents: List[HeterogeneousAgent],
    x_i: np.ndarray,
    gat: EpistemicGAT,
    belief_fuser: Any,
    config: Any,
) -> float:
    """
    Run M1 -> M4 -> M2 on a single sample without calibration or decision.
    Used to produce collective probabilities for conformal calibration.
    """
    # M1: predictions + UQ
    predictions = {}
    uncertainties = {}
    for agent in agents:
        p = agent.predict(x_i)
        uq = agent.estimate_uncertainty(x_i)
        predictions[agent.agent_id] = p
        uncertainties[agent.agent_id] = {
            "epistemic": uq.epistemic,
            "aleatoric": uq.aleatoric,
            "total": uq.total,
        }

    # Trust weights
    eps = 1e-8
    raw_w = {a: 1.0 / (uncertainties[a]["epistemic"] + eps) for a in predictions}
    total_w = sum(raw_w.values())
    trust_weights = {a: raw_w[a] / total_w for a in raw_w}

    # M4: EpistemicGAT
    agent_order = list(predictions.keys())
    N = len(agent_order)
    node_features = np.array([
        [predictions[a], uncertainties[a]["epistemic"], uncertainties[a]["aleatoric"]]
        for a in agent_order
    ], dtype=np.float32)
    adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
    epi_unc = {a: uncertainties[a]["epistemic"] for a in agent_order}

    H_prime, _ = gat.forward(node_features, adjacency, epi_unc, agent_order)
    refined_probs = 1.0 / (1.0 + np.exp(-H_prime[:, 0]))  # sigmoid mapping
    for idx, a in enumerate(agent_order):
        predictions[a] = float(np.clip(refined_probs[idx], 1e-7, 1 - 1e-7))

    # M2: Belief Fusion
    from cognix.belief.base import BeliefState, FusionStrategy
    beliefs = []
    for a, p in predictions.items():
        beliefs.append(BeliefState(
            agent_id=a,
            belief=np.array([1 - p, p]),
            alpha=p * 10,
            beta_param=(1 - p) * 10,
            confidence=p,
        ))

    fused = belief_fuser.fuse(
        beliefs=beliefs,
        strategy=FusionStrategy.EPISTEMIC_WEIGHTED,
        epistemic_uncertainties={a: uncertainties[a]["epistemic"] for a in uncertainties},
        reliabilities={a: 1.0 for a in uncertainties},
    )
    return float(np.clip(fused.confidence, 0.0, 1.0))


# ==========================================
# 4. EVALUATION PROTOCOL
# ==========================================
def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mode", type=str, default="research",
                        choices=["research", "production"])
    args = parser.parse_args()

    n_samples = args.samples
    pipeline_mode = args.mode
    seeds = [42, 101]
    conditions = ["NORMAL", "HIGH_NOISE", "OOD_SHIFT", "MISSING_AGENT"]
    baselines = ["uniform", "confidence", "bayesian", "epistemic_weighted", "standard_gat"]

    run_id = f"EXP-RQ0-{int(time.time())}"
    results_dir = os.path.join("results", "RQ_SYNTHETIC_001")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nCOGNIX RQ_SYNTHETIC_001 | mode={pipeline_mode} | "
          f"samples={n_samples} | seeds={seeds}")
    print("=" * 65)

    all_results = []
    dashboard_traces = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        torch.manual_seed(seed)

        # ── 3-way split: 60% train / 20% calibration / 20% test ──────
        gen = DataGenerator(n_samples=n_samples, seed=seed)
        X_all, y_all = gen.generate_base()

        n_train = int(0.6 * n_samples)
        n_cal = int(0.2 * n_samples)
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_cal,   y_cal   = X_all[n_train:n_train + n_cal], y_all[n_train:n_train + n_cal]
        X_base_test, y_base_test = X_all[n_train + n_cal:], y_all[n_train + n_cal:]

        # ── Build agents ──────────────────────────────────────────────
        agents = [
            HeterogeneousAgent("Agent_A_Feat0", FeatureModel(0)),
            HeterogeneousAgent("Agent_B_Feat1", FeatureModel(1)),
            HeterogeneousAgent("Agent_C_Feat2", FeatureModel(2)),
            HeterogeneousAgent("Agent_D_AllFeat", MultiFeatureModel()),
        ]

        print(f"  Training agents on {len(X_train)} samples...")
        for agent in agents:
            agent.fit(X_train, y_train, epochs=20)

        # ── Phase 5: Compute per-agent reliability on calibration set ─
        # Reliability = per-agent classification accuracy on X_cal.
        # Defined as: r_i = mean(round(p_i(x)) == y(x)) for x in X_cal
        agent_reliabilities = {}
        for agent in agents:
            agent.healthy = True
            preds_cal = [round(agent.predict(X_cal[i]).value) for i in range(len(X_cal))]
            agent_reliabilities[agent.agent_id] = float(
                np.mean(np.array(preds_cal) == y_cal)
            )
        print(f"  Agent reliabilities (cal accuracy): {agent_reliabilities}")

        # ── Train EpistemicGAT on X_train ─────────────────────────────
        # GAT parameters are trained end-to-end on training data only.
        # Training objective: BCE(mean(sigmoid(H_prime[:,0])), y_train)
        # This is required because random W produces arbitrary outputs.
        # Calibration and test data are NOT used here.
        shared_gat = EpistemicGAT(
            num_layers=2, input_dim=3, hidden_dim=8, output_dim=4, use_epistemic_prior=True
        )
        torch.manual_seed(seed)
        gat_losses = shared_gat.fit(
            agents=agents, X_train=X_train, y_train=y_train, epochs=20, lr=1e-3, verbose=False, seed=seed,
        )
        print(f"  GAT (Epistemic) trained: initial_loss={gat_losses[0]:.4f}  "
              f"final_loss={gat_losses[-1]:.4f}")

        # Train Standard GAT for Ablation
        standard_gat = EpistemicGAT(
            num_layers=2, input_dim=3, hidden_dim=8, output_dim=4, use_epistemic_prior=False
        )
        torch.manual_seed(seed)
        gat_standard_losses = standard_gat.fit(
            agents=agents, X_train=X_train, y_train=y_train, epochs=20, lr=1e-3, verbose=False, seed=seed,
        )
        print(f"  GAT (Standard) trained: initial_loss={gat_standard_losses[0]:.4f}  "
              f"final_loss={gat_standard_losses[-1]:.4f}")

        shared_fuser = EpistemicWeightedFusion(eps=1e-8)
        config = CognixConfig()

        if not shared_gat.is_trained():
            raise RuntimeError("Research mode: GAT must be trained before calibration.")

        # ── Phase 8: Pre-fit conformal predictor on collective outputs ─
        # Calibration predictions must come from the SAME pipeline as inference:
        # M1 -> M4 (GAT) -> M2 (Fusion) -> collective probability
        print(f"  Calibrating conformal predictor on {len(X_cal)} samples...")
        cal_collective_probs = []
        for i in range(len(X_cal)):
            p_col = run_pre_calibration(
                agents, X_cal[i], shared_gat, shared_fuser, config
            )
            cal_collective_probs.append([1.0 - p_col, p_col])

        shared_calibrator = ConformalPredictor()
        shared_calibrator.fit(
            cal_outputs=np.array(cal_collective_probs),
            cal_labels=y_cal.astype(int),
        )
        print(f"  Calibrator (Epistemic) fitted: n_cal={shared_calibrator.n_cal}, "
              f"q_hat={shared_calibrator.cal_scores[-1]:.4f}")

        # Calibrate Standard GAT
        print(f"  Calibrating standard conformal predictor...")
        cal_standard_probs = []
        for i in range(len(X_cal)):
            p_col = run_pre_calibration(
                agents, X_cal[i], standard_gat, shared_fuser, config
            )
            cal_standard_probs.append([1.0 - p_col, p_col])

        standard_calibrator = ConformalPredictor()
        standard_calibrator.fit(
            cal_outputs=np.array(cal_standard_probs),
            cal_labels=y_cal.astype(int),
        )
        print(f"  Calibrator (Standard) fitted: n_cal={standard_calibrator.n_cal}, "
              f"q_hat={standard_calibrator.cal_scores[-1]:.4f}")

        # ── Test across Conditions ────────────────────────────────────
        for cond in conditions:
            X_test, y_test, _ = gen.apply_condition(
                X_base_test, y_base_test, cond
            )

            for agent in agents:
                agent.healthy = True
            if cond == "MISSING_AGENT":
                agents[1].healthy = False

            # ── Test across Baselines ─────────────────────────────────
            for bline in baselines:
                current_gat = standard_gat if bline == "standard_gat" else shared_gat
                current_calibrator = standard_calibrator if bline == "standard_gat" else shared_calibrator
                engine_belief = "epistemic_weighted" if bline == "standard_gat" else bline

                engine = DecisionEngine(
                    config=config,
                    belief=engine_belief,
                    attribution="epistemic_shapley",
                    graph=current_gat,
                    calibrator=current_calibrator,
                    mode=pipeline_mode,
                )
                # Inject the pre-fitted calibrator directly
                engine.pipeline.calibrator = current_calibrator

                tracker = LatencyTracker()
                preds = []
                all_pred_sets = []

                for i in range(len(X_test)):
                    x_i = X_test[i]
                    t0 = time.perf_counter()
                    try:
                        res = engine.decide(
                            agents, x_i,
                            agent_reliabilities=agent_reliabilities,
                        )
                    except RuntimeError as e:
                        print(f"  RESEARCH MODE FAIL at {cond}/{bline}/sample_{i}: {e}")
                        sys.exit(1)

                    tracker.record("total", (time.perf_counter() - t0) * 1000)
                    preds.append(res.confidence)

                    # Capture calibration prediction set for coverage
                    cal_info = res.calibration_metrics or {}
                    ps = cal_info.get("prediction_set", [])
                    all_pred_sets.append(ps)

                    # Save trace for dashboard (seed=42, EWF only)
                    if seed == 42 and bline == "epistemic_weighted":
                        mod_status = res.metadata.get("module_status", {}) if res.metadata else {}
                        trace = DecisionTrace(
                            timestamp=time.time(),
                            input_id=f"{cond}_sample_{i}",
                            agent_predictions=res.agent_predictions,
                            uncertainty={
                                "epistemic": res.epistemic_uncertainty,
                                "aleatoric": res.aleatoric_uncertainty,
                                "total": res.total_uncertainty,
                            },
                            belief=[1 - res.confidence, res.confidence],
                            weights=res.agent_trust_weights,
                            communication=res.communication_statistics or {},
                            calibration=res.calibration_metrics or {},
                            risk=res.risk_level.value,
                            decision=res.decision.value,
                            attribution=res.agent_contributions,
                            latency=res.latency_ms,
                            module_status=mod_status,
                        )
                        dashboard_traces.append(trace.to_dict())

                # ── Metrics ───────────────────────────────────────────
                ece = calculate_ece(np.array(preds), y_test)
                acc = accuracy(np.array(preds), y_test)
                brier = brier_score(np.array(preds), y_test)
                lat = tracker.get_percentiles("total")

                # Conformal coverage (for epistemic_weighted and standard_gat only)
                coverage = None
                if bline in ["epistemic_weighted", "standard_gat"] and all_pred_sets:
                    # prediction_set contains class indices; y=1 means OBSTACLE (class 1)
                    covered = sum(
                        int(y_test[i]) in all_pred_sets[i]
                        for i in range(len(y_test))
                        if all_pred_sets[i]
                    )
                    valid = sum(1 for ps in all_pred_sets if ps)
                    coverage = covered / valid if valid > 0 else None
                    if coverage is not None:
                        print(f"  [{cond}/{bline}] ECE={ece:.4f} "
                              f"Acc={acc:.4f} Coverage={coverage:.4f}")
                    else:
                        print(f"  [{cond}/{bline}] ECE={ece:.4f} Acc={acc:.4f}")
                else:
                    print(f"  [{cond}/{bline}] ECE={ece:.4f} Acc={acc:.4f}")

                all_results.append({
                    "seed": seed,
                    "condition": cond,
                    "baseline": bline,
                    "ece": ece,
                    "accuracy": acc,
                    "brier": brier,
                    "latency_p95": lat["p95"],
                    "coverage": coverage,
                })

    # ── Aggregation & Export ──────────────────────────────────────────
    aggregated = {}
    for res in all_results:
        key = f"{res['condition']}_{res['baseline']}"
        if key not in aggregated:
            aggregated[key] = {
                "ece": [], "acc": [], "brier": [], "lat": [], "coverage": []
            }
        aggregated[key]["ece"].append(res["ece"])
        aggregated[key]["acc"].append(res["accuracy"])
        aggregated[key]["brier"].append(res["brier"])
        aggregated[key]["lat"].append(res["latency_p95"])
        if res["coverage"] is not None:
            aggregated[key]["coverage"].append(res["coverage"])

    # Build dashboard metrics
    dash_metrics = {
        "cognix_ece": {
            "value": float(np.mean(
                aggregated.get("OOD_SHIFT_epistemic_weighted", {}).get("ece", [0.0])
            ))
        },
        "baseline_ece": {
            "value": float(np.mean(
                aggregated.get("OOD_SHIFT_uniform", {}).get("ece", [0.0])
            ))
        },
        "cognix_acc": float(np.mean(
            aggregated.get("OOD_SHIFT_epistemic_weighted", {}).get("acc", [0.0])
        )),
        "baseline_acc": float(np.mean(
            aggregated.get("OOD_SHIFT_uniform", {}).get("acc", [0.0])
        )),
    }

    if dashboard_traces:
        dashboard_traces[-1]["metrics"] = dash_metrics
        with open(os.path.join(results_dir, "trace.json"), "w") as f:
            json.dump(dashboard_traces, f, indent=2)
        with open(os.path.join("results", "latest_run.txt"), "w") as f:
            f.write("RQ_SYNTHETIC_001")

    # Save final metrics
    final_metrics = []
    for key, vals in aggregated.items():
        parts = key.split("_", 1)
        cond = parts[0]
        bline = parts[1] if len(parts) > 1 else key
        row = {
            "condition": cond,
            "baseline": bline,
            "ece_mean": float(np.mean(vals["ece"])),
            "ece_std": float(np.std(vals["ece"])),
            "acc_mean": float(np.mean(vals["acc"])),
            "acc_std": float(np.std(vals["acc"])),
            "brier_mean": float(np.mean(vals["brier"])),
            "brier_std": float(np.std(vals["brier"])),
            "latency_p95_mean": float(np.mean(vals["lat"])),
        }
        if vals["coverage"]:
            row["coverage_mean"] = float(np.mean(vals["coverage"]))
        final_metrics.append(row)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    meta = ExperimentMetadata(
        run_id=run_id,
        dataset="SYNTHETIC_SHIFT_3D",
        dataset_version="2.0",
        model="Heterogeneous_MCDropout_Ensemble_GAT_Conformal",
        model_version="2.0",
        experiment="RQ_SYNTHETIC_001",
        seed=0,
        configuration={
            "n_samples": n_samples,
            "seeds": seeds,
            "conditions": conditions,
            "mode": pipeline_mode,
            "gat_layers": 2,
            "gat_hidden": 8,
            "mc_dropout_T": 30,
            "alpha": 0.05,
        }
    )
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"Execution Complete. Results saved to {results_dir}/")
    print(f"Run ID: {run_id}")
    print(f"Pipeline mode: {pipeline_mode}")
    print("\nKey OOD_SHIFT results (mean over 5 seeds):")
    for row in final_metrics:
        if row["condition"] == "OOD_SHIFT":
            cov = f"  coverage={row.get('coverage_mean', 'N/A'):.4f}" \
                if row.get("coverage_mean") else ""
            print(f"  {row['baseline']:25s} ECE={row['ece_mean']:.4f} "
                  f"Acc={row['acc_mean']:.4f}{cov}")


if __name__ == "__main__":
    run_evaluation()
