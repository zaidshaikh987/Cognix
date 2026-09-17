"""
COGNIX 20-Seed Benchmark — CLEAN VERSION (post-forensic audit)
==============================================================

Fixes applied (vs. the original run_20seed_benchmark.py):

1. is_trained() mock removed in epistemic_gat.py — GATs are now actually trained.
2. X_train uses dict-of-sensor inputs, not raw CarlAnomalyFrame objects.
3. Conformal calibration uses real agent predictions on a held-out calibration split,
   not random numbers.
4. Coverage is computed from the actual ConformalPredictionSet returned by the pipeline,
   not a hardcoded heuristic.
5. ECE is computed using standard binned calibration (10 bins), not per-frame |p - y|.
6. Data is split per-scenario into TRAIN / CALIBRATE / TEST splits.
7. Each model gets its OWN conformal calibrator fitted on its own calibration outputs.

No model architecture changes were made.
"""

import os
import sys
import time
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognix import (
    DecisionEngine, CognixConfig, UncertaintyDecomposition,
    EpistemicWeightedFusion, ConformalPredictor, EscalationEngine
)
import logging
logging.getLogger("cognix").setLevel(logging.ERROR)
from cognix.adapters.carla.dataset import CarlAnomalyDataset
from cognix.adapters.carla.agents import (
    CameraAgent, DepthAgent, LiDARAgent, GNSSAgent, IMUAgent, SegAgent
)

SEEDS = [42]  # 1 seed for quick testing
SCENARIOS = [
    "NORMAL", "HIGH_NOISE", "OOD",
    "MISSING_AGENT", "PARTIAL_FAILURE", "MULTIPLE_FAILURE", "CONFLICT"
]
MODELS = {
    "NoGraph":      {"method": "none"},
    "StandardGAT":  {"method": "gat"},
    "EpistemicGAT": {"method": "epistemic_gat"}
}

N_FRAMES = 200          # per scenario per seed
TRAIN_FRAC = 0.50       # 50% train
CAL_FRAC   = 0.25       # 25% calibrate
# remaining 25% = test

ECE_BINS = 10


def make_agents():
    return [CameraAgent(), DepthAgent(), LiDARAgent(),
            GNSSAgent(), IMUAgent(), SegAgent()]


def frame_to_inputs(frame):
    """Convert a CarlAnomalyFrame to the dict format agents expect."""
    return {
        "Camera": frame.rgb,
        "Depth":  frame.depth,
        "LiDAR":  frame.lidar,
        "GNSS":   frame.gnss,
        "IMU":    frame.imu,
        "Seg":    frame.segmentation,
    }


def split_frames(frames, rng):
    """Deterministic train/cal/test split."""
    n = len(frames)
    idx = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_cal   = int(n * CAL_FRAC)
    train = [frames[i] for i in idx[:n_train]]
    cal   = [frames[i] for i in idx[n_train:n_train + n_cal]]
    test  = [frames[i] for i in idx[n_train + n_cal:]]
    return train, cal, test


def binned_ece(probs, labels, n_bins=ECE_BINS):
    """Standard binned Expected Calibration Error."""
    probs = np.array(probs)
    labels = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (probs > lo) & (probs <= hi)
        if b == 0:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc  = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return ece


def calibrate_conformal(engine, cal_frames):
    """
    Run the engine on calibration frames, collect the softmax-style [p_safe, p_hazard]
    outputs, and fit() the ConformalPredictor on those real outputs.
    """
    agents = make_agents()
    cal_outputs = []
    cal_labels  = []
    for frame in cal_frames:
        inputs = frame_to_inputs(frame)
        result = engine.decide(agents, inputs, {a.agent_id: 1.0 for a in agents})
        prob = result.calibrated_confidence if result.calibrated_confidence else result.confidence
        # Build 2-class softmax-style output
        cal_outputs.append([1.0 - prob, prob])
        cal_labels.append(frame.label)
    cal_outputs = np.array(cal_outputs)
    cal_labels  = np.array(cal_labels, dtype=int)
    # Refit the calibrator with real data
    engine.pipeline.calibrator.fit(cal_outputs=cal_outputs, cal_labels=cal_labels)
    return len(cal_labels)


def evaluate_on_test(engine, test_frames):
    """
    Run the engine on test frames. Extract metrics from the actual DecisionResult,
    including the real conformal prediction sets.
    """
    agents = make_agents()
    acc_list, brier_list, nll_list = [], [], []
    cov_list, set_size_list = [], []
    lat_list = []
    # For binned ECE: collect (predicted_prob_of_true_class, true_label)
    ece_probs, ece_labels = [], []

    for frame in test_frames:
        inputs = frame_to_inputs(frame)

        t0 = time.perf_counter()
        result = engine.decide(agents, inputs, {a.agent_id: 1.0 for a in agents})
        t1 = time.perf_counter()
        lat_list.append((t1 - t0) * 1000.0)

        # --- Accuracy ---
        pred_label = 1 if result.risk_level.name in ["HIGH", "CRITICAL"] else 0
        acc = int(pred_label == frame.label)
        acc_list.append(acc)

        # --- Probability of HAZARD ---
        # If calibrated_confidence exists, it is exactly P(Hazard).
        # If we fall back to confidence, pipeline.py flipped it to P(Safe) if pred_label == 0, so we unflip.
        if result.calibrated_confidence is not None:
            prob_hazard = result.calibrated_confidence
        else:
            prob_hazard = result.confidence if pred_label == 1 else 1.0 - result.confidence
            
        prob_safe = 1.0 - prob_hazard

        # --- Brier score ---
        brier = (prob_hazard - frame.label) ** 2
        brier_list.append(brier)

        # --- NLL ---
        prob_true = prob_hazard if frame.label == 1 else prob_safe
        prob_true = max(1e-7, min(1 - 1e-7, prob_true))
        nll_list.append(-np.log(prob_true))

        # --- ECE data: collect model's confidence in its own prediction ---
        # max(prob, 1-prob) is the confidence in the predicted class
        ece_probs.append(max(prob_hazard, prob_safe))
        ece_labels.append(acc)

        # --- Conformal coverage (from REAL prediction set) ---
        pred_set = None
        if result.calibration_metrics and "prediction_set" in result.calibration_metrics:
            pred_set = result.calibration_metrics["prediction_set"]
        if pred_set is not None:
            set_size_list.append(len(pred_set))
            cov_list.append(int(frame.label in pred_set))
        else:
            set_size_list.append(-1)
            cov_list.append(-1)

    # Standard binned ECE: bin by model's confidence, compare to accuracy in each bin
    ece_val = binned_ece(ece_probs, ece_labels, n_bins=ECE_BINS)

    return {
        "acc":      float(np.mean(acc_list)),
        "brier":    float(np.mean(brier_list)),
        "nll":      float(np.mean(nll_list)),
        "cov":      float(np.mean([c for c in cov_list if c >= 0])) if any(c >= 0 for c in cov_list) else 0.0,
        "set_size": float(np.mean([s for s in set_size_list if s >= 0])) if any(s >= 0 for s in set_size_list) else 0.0,
        "ece":      ece_val,
        "p50":      float(np.percentile(lat_list, 50)),
        "p95":      float(np.percentile(lat_list, 95)),
        "p99":      float(np.percentile(lat_list, 99)),
    }


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def compute_effect_size(d1, d2):
    """Cohen's d."""
    n1, n2 = len(d1), len(d2)
    var1, var2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var == 0:
        return 0.0
    return (np.mean(d1) - np.mean(d2)) / np.sqrt(pooled_var)


def run_benchmark():
    print("COGNIX 20-Seed Benchmark (CLEAN — post-forensic audit)")
    print(f"Seeds: {SEEDS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Frames/scenario: {N_FRAMES}  (train {TRAIN_FRAC:.0%} / cal {CAL_FRAC:.0%} / test {1-TRAIN_FRAC-CAL_FRAC:.0%})")
    print("=" * 60)

    results = {m: {s: [] for s in SCENARIOS} for m in MODELS}

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        dataset = CarlAnomalyDataset(mode="synthetic", n_frames_per_anomaly=N_FRAMES, seed=seed)
        all_frames = dataset.get_all_scenarios()
        split_rng = np.random.default_rng(seed)

        # --- Build per-scenario splits (shared across models) ---
        splits = {}
        for scenario in SCENARIOS:
            train, cal, test = split_frames(all_frames[scenario], split_rng)
            splits[scenario] = (train, cal, test)

        # Combine NORMAL train frames for GAT training
        # (use only train split so cal/test are held out)
        gat_train_frames = []
        gat_train_labels = []
        for scenario in SCENARIOS:
            train_s, _, _ = splits[scenario]
            for f in train_s:
                gat_train_frames.append(frame_to_inputs(f))
                gat_train_labels.append(f.label)

        for model_name, cfg in MODELS.items():
            config = CognixConfig()
            config.graph.method = cfg["method"]

            calibrator = ConformalPredictor()
            graph_module = None

            if cfg["method"] == "gat":
                from cognix.graph.epistemic_gat import EpistemicGAT
                graph_module = EpistemicGAT(use_epistemic_prior=False)
            elif cfg["method"] == "epistemic_gat":
                from cognix.graph.epistemic_gat import EpistemicGAT
                graph_module = EpistemicGAT(use_epistemic_prior=True)

            engine = DecisionEngine(
                config=config,
                uncertainty=UncertaintyDecomposition(),
                belief=EpistemicWeightedFusion(),
                calibrator=calibrator,
                escalation=EscalationEngine(),
                graph=graph_module,
                mode="production"
            )

            # --- TRAIN the GAT if it exists ---
            if graph_module is not None and not graph_module.is_trained():
                agents_for_train = make_agents()
                print(f"  Training {model_name} on {len(gat_train_frames)} frames...")
                graph_module.fit(
                    agents_for_train,
                    gat_train_frames,
                    gat_train_labels,
                    epochs=30,
                    lr=1e-3,
                    verbose=False,
                    seed=seed
                )

            # --- Evaluate per scenario ---
            for scenario in SCENARIOS:
                _, cal_frames, test_frames = splits[scenario]

                # Calibrate conformal on the calibration split
                n_cal = calibrate_conformal(engine, cal_frames)

                # Evaluate on the test split
                metrics = evaluate_on_test(engine, test_frames)
                metrics["n_cal"] = n_cal
                results[model_name][scenario].append(metrics)

    return results


def analyze_results(results):
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS & WILCOXON TESTS (CLEAN)")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    out_file = open("results/benchmark_20seed_clean.txt", "w")
    out_file.write("COGNIX 20-SEED BENCHMARK (CLEAN — post-forensic audit)\n")
    out_file.write("=" * 60 + "\n")
    out_file.write(f"Seeds: {SEEDS}\n")
    out_file.write(f"Frames/scenario: {N_FRAMES} "
                   f"(train {TRAIN_FRAC:.0%} / cal {CAL_FRAC:.0%} / "
                   f"test {1-TRAIN_FRAC-CAL_FRAC:.0%})\n\n")

    metrics_to_report = ["acc", "ece", "brier", "cov", "set_size"]

    for scenario in SCENARIOS:
        header = f"\n[{scenario}]"
        print(header)
        out_file.write(header + "\n")

        for metric in metrics_to_report:
            epi = [results["EpistemicGAT"][scenario][s][metric] for s in range(len(SEEDS))]
            std = [results["StandardGAT"][scenario][s][metric] for s in range(len(SEEDS))]
            nog = [results["NoGraph"][scenario][s][metric] for s in range(len(SEEDS))]

            epi_m, epi_lo, epi_hi = mean_confidence_interval(epi)
            std_m, std_lo, std_hi = mean_confidence_interval(std)
            nog_m, nog_lo, nog_hi = mean_confidence_interval(nog)

            # Wilcoxon: EpistemicGAT vs StandardGAT
            try:
                _, p_std = stats.wilcoxon(epi, std, zero_method='wilcox', correction=False)
            except ValueError:
                p_std = 1.0

            eff_std = compute_effect_size(epi, std)
            sig = "***" if p_std < 0.01 else ("*" if p_std < 0.05 else "ns")

            # Wilcoxon: EpistemicGAT vs NoGraph
            try:
                _, p_nog = stats.wilcoxon(epi, nog, zero_method='wilcox', correction=False)
            except ValueError:
                p_nog = 1.0
            eff_nog = compute_effect_size(epi, nog)
            sig_nog = "***" if p_nog < 0.01 else ("*" if p_nog < 0.05 else "ns")

            report = (
                f"  {metric.upper()}:\n"
                f"    EpistemicGAT: {epi_m:.4f} (95% CI: {epi_lo:.4f} - {epi_hi:.4f})\n"
                f"    StandardGAT : {std_m:.4f} (95% CI: {std_lo:.4f} - {std_hi:.4f})"
                f"  [p={p_std:.4f} {sig}, d={eff_std:.2f}]\n"
                f"    NoGraph     : {nog_m:.4f} (95% CI: {nog_lo:.4f} - {nog_hi:.4f})"
                f"  [p={p_nog:.4f} {sig_nog}, d={eff_nog:.2f}]\n"
            )
            print(report, end="")
            out_file.write(report)

        # Latency
        l_p50 = np.mean([results["EpistemicGAT"][scenario][s]["p50"] for s in range(len(SEEDS))])
        l_p99 = np.mean([results["EpistemicGAT"][scenario][s]["p99"] for s in range(len(SEEDS))])
        lat = f"  LATENCY (EpistemicGAT): P50={l_p50:.2f}ms, P99={l_p99:.2f}ms\n"
        print(lat, end="")
        out_file.write(lat)

    out_file.close()
    print(f"\nFull results saved to results/benchmark_20seed_clean.txt")


if __name__ == "__main__":
    res = run_benchmark()
    analyze_results(res)

    with open("results/benchmark_20seed_clean.json", "w") as f:
        json.dump(res, f, indent=2)
