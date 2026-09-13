"""
COGNIX Ground-Truth Verification Suite
======================================

Every test in this file:
  1. Has an expected value that can be derived BY HAND (shown in the docstring).
  2. Is mathematically independent of the COGNIX codebase.
  3. PASSES or FAILS — no interpretation needed.

Run with:
    python tests/test_ground_truth.py

If EVERY test passes, the core mathematical primitives are correct.
If any test fails, that component is broken.
"""

import sys
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, condition))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ECE — Expected Calibration Error
# Formula: ECE = Σ_m (|B_m|/n) * |mean(y_true in B_m) - mean(p_hat in B_m)|
# ══════════════════════════════════════════════════════════════════════════════

from cognix.metrics.evaluation import calculate_ece

print("\n" + "═"*70)
print("SECTION 1: ECE — Expected Calibration Error")
print("═"*70)

# ── Test 1.1 ──────────────────────────────────────────────────────────────
# By hand:
#   All predictions = 0.3.  All labels = 0 or 1 in exactly 30/70 ratio.
#   One bin [0.2, 0.3):  mean_conf = 0.3,  frac_positives = 0.3
#   ECE = 1.0 * |0.3 - 0.3| = 0.0
preds = np.full(100, 0.3)
labels = np.array([1.0]*30 + [0.0]*70)
ece = calculate_ece(preds, labels, bins=10)
check("ECE-1.1: uniform p=0.3, 30% positives → ECE=0.0",
      abs(ece) < 0.001,
      f"got {ece:.6f}, expected 0.000")

# ── Test 1.2 ──────────────────────────────────────────────────────────────
# By hand:
#   All predictions = 0.9.  All labels = 0 (0% positive).
#   One bin [0.9, 1.0]:  mean_conf = 0.9,  frac_positives = 0.0
#   ECE = 1.0 * |0.0 - 0.9| = 0.9
preds = np.full(100, 0.9)
labels = np.zeros(100)
ece = calculate_ece(preds, labels, bins=10)
check("ECE-1.2: p=0.9, 0% positives → ECE=0.9",
      abs(ece - 0.9) < 0.001,
      f"got {ece:.6f}, expected 0.900")

# ── Test 1.3 ──────────────────────────────────────────────────────────────
# By hand:
#   50 samples with p=0.8, all label=1 → bin [0.7,0.8): conf=0.8, frac=1.0, gap=0.2
#   50 samples with p=0.2, all label=0 → bin [0.2,0.3): conf=0.2, frac=0.0, gap=0.2
#   ECE = 0.5*0.2 + 0.5*0.2 = 0.2
preds = np.array([0.8]*50 + [0.2]*50)
labels = np.array([1.0]*50 + [0.0]*50)
ece = calculate_ece(preds, labels, bins=10)
check("ECE-1.3: bimodal p=[0.8,0.2], correct labels → ECE=0.2",
      abs(ece - 0.2) < 0.001,
      f"got {ece:.6f}, expected 0.200")

# ── Test 1.4 ──────────────────────────────────────────────────────────────
# By hand: empty → ECE = 0.0
ece = calculate_ece(np.array([]), np.array([]))
check("ECE-1.4: empty input → ECE=0.0",
      ece == 0.0,
      f"got {ece}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: NoGraph — logit identity round-trip
# sigmoid(logit(p)) = p  exactly (within float32 precision)
# ══════════════════════════════════════════════════════════════════════════════

from cognix.graph.no_graph import NoGraph

print("\n" + "═"*70)
print("SECTION 2: NoGraph — logit/sigmoid identity")
print("═"*70)

ng = NoGraph()
adj = np.ones((3, 3)) - np.eye(3)

# ── Test 2.1 ──────────────────────────────────────────────────────────────
# By hand:
#   logit(0.7) = log(0.7/0.3) = 0.8473
#   sigmoid(0.8473) = 1/(1+e^-0.8473) = 0.7000
test_probs = [0.7, 0.3, 0.9]
nf = np.array([[p, 0.001, 0.1] for p in test_probs], dtype=np.float32)
res = ng.forward(nf, adj, {}, ['a','b','c'])
sigmoid_out = 1.0 / (1.0 + np.exp(-res.node_outputs[:, 0]))
max_err = float(np.max(np.abs(sigmoid_out - np.array(test_probs))))
check("NoGraph-2.1: sigmoid(logit(p)) == p for [0.7, 0.3, 0.9]",
      max_err < 1e-5,
      f"max abs error: {max_err:.2e}, expected <1e-5")

# ── Test 2.2 ──────────────────────────────────────────────────────────────
# By hand: logit(0.5) = 0 → sigmoid(0) = 0.5
nf2 = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
adj2 = np.array([[1.0]])
res2 = ng.forward(nf2, adj2, {}, ['a'])
out = float(1.0 / (1.0 + np.exp(-res2.node_outputs[0, 0])))
check("NoGraph-2.2: logit(0.5)=0, sigmoid(0)=0.5",
      abs(out - 0.5) < 1e-6,
      f"got {out:.8f}, expected 0.5")

# ── Test 2.3 ──────────────────────────────────────────────────────────────
# Works with Tensor input (the crash we fixed)
import torch
nf_tensor = torch.tensor([[0.6, 0.001, 0.1], [0.4, 0.002, 0.15]], dtype=torch.float32)
adj_np = np.ones((2, 2)) - np.eye(2)
try:
    res3 = ng.forward(nf_tensor, adj_np, {}, ['a','b'])
    out3 = 1.0 / (1.0 + np.exp(-res3.node_outputs[:, 0]))
    err3 = float(np.max(np.abs(out3 - np.array([0.6, 0.4]))))
    check("NoGraph-2.3: accepts torch.Tensor input without crash",
          err3 < 1e-5,
          f"max abs error: {err3:.2e}")
except Exception as e:
    check("NoGraph-2.3: accepts torch.Tensor input without crash", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Conformal Predictor — coverage guarantee
# Theory: with n_cal calibration samples, empirical coverage >= 1-alpha
# ══════════════════════════════════════════════════════════════════════════════

from cognix.calibration.conformal import ConformalPredictor

print("\n" + "═"*70)
print("SECTION 3: Conformal Predictor")
print("═"*70)

# ── Test 3.1 ──────────────────────────────────────────────────────────────
# By hand:
#   100 calibration samples all class 1, predicted with p=0.9 → [0.1, 0.9]
#   nonconformity score = 1 - p(true_class) = 1 - 0.9 = 0.1 for all
#   Sorted cal scores = [0.1]*100
#   q_idx = ceil(101 * 0.95) = ceil(95.95) = 96
#   q_val = cal_scores[95] = 0.1
#   For test sample [0.0, 1.0]: score_class0 = 1-0.0=1.0 > 0.1 → not included
#                                score_class1 = 1-1.0=0.0 <= 0.1 → included
#   Prediction set = {1}
rng = np.random.default_rng(42)
n_cal = 100
cal_probs = np.column_stack([
    np.full(n_cal, 0.1),
    np.full(n_cal, 0.9)
])
cal_labels = np.ones(n_cal, dtype=int)
cp = ConformalPredictor()
cp.fit(cal_probs, cal_labels)
test_probs = np.array([[0.1, 0.9]])
ps = cp.predict(test_probs, alpha=0.05)
check("ConformalPredictor-3.1: high-confidence class-1 → pred set = {1}",
      ps[0].prediction_set == [1],
      f"got {ps[0].prediction_set}, expected [1]")

# ── Test 3.2 ──────────────────────────────────────────────────────────────
# By hand (coverage guarantee):
#   For inductive conformal prediction with n_cal calibration samples:
#   Finite-sample guarantee: E[coverage] >= 1 - alpha  (in expectation over seeds)
#   We average over 10 independent seeds to reduce single-seed sampling variance.
#   With n_cal=500, alpha=0.10, n_test=1000: std of coverage ~ 0.009 per seed.
#   Average of 10 seeds has std ~ 0.003, so mean >= 0.897 with very high probability.
coverages = []
for seed_i in range(10):
    rng_i = np.random.default_rng(seed_i)
    n_cal_i, n_test_i = 500, 1000
    all_s_i = rng_i.uniform(0, 1, n_cal_i + n_test_i)
    cal_s_i, test_s_i = all_s_i[:n_cal_i], all_s_i[n_cal_i:]
    cal_probs_i  = np.column_stack([1 - cal_s_i,  cal_s_i])
    test_probs_i = np.column_stack([1 - test_s_i, test_s_i])
    y_cal_i  = np.zeros(n_cal_i,  dtype=int)
    y_test_i = np.zeros(n_test_i, dtype=int)
    cp_i = ConformalPredictor()
    cp_i.fit(cal_probs_i, y_cal_i)
    ps_i = cp_i.predict(test_probs_i, alpha=0.10)
    cov_i = sum(1 for j, p in enumerate(ps_i) if y_test_i[j] in p.prediction_set) / n_test_i
    coverages.append(cov_i)
mean_cov = float(np.mean(coverages))
check("ConformalPredictor-3.2: mean coverage >= 0.895 averaged over 10 seeds (alpha=0.10)",
      mean_cov >= 0.895,
      f"mean coverage={mean_cov:.4f} over seeds, expected >= 0.895. Per-seed: {[round(c,3) for c in coverages]}")





# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EpistemicWeightedFusion — math
# w_j = rel_j / (eps_j + eps), normalized
# ══════════════════════════════════════════════════════════════════════════════

from cognix.belief.fusion import EpistemicWeightedFusion

print("\n" + "═"*70)
print("SECTION 4: EpistemicWeightedFusion")
print("═"*70)

# ── Test 4.1 ──────────────────────────────────────────────────────────────
# By hand:
#   All reliabilities = 1, equal uncertainties → uniform weights → simple average
#   predictions = {A: 0.8, B: 0.4} → fused = 0.6
fuser = EpistemicWeightedFusion(eps=1e-6)
res_f = fuser.fuse(
    predictions={"A": 0.8, "B": 0.4},
    uncertainties={"A": 1.0, "B": 1.0},
    reliabilities={"A": 1.0, "B": 1.0}
)
check("Fusion-4.1: equal weights → simple average → (0.8+0.4)/2 = 0.6",
      abs(res_f.probability - 0.6) < 1e-4,
      f"got {res_f.probability:.6f}, expected 0.6")

# ── Test 4.2 ──────────────────────────────────────────────────────────────
# By hand:
#   Agent A: rel=1, unc=0.001 → w_A = 1/0.001001 ≈ 999
#   Agent B: rel=1, unc=1.0   → w_B = 1/1.000001 ≈ 1.0
#   total ≈ 1000, w_A_norm ≈ 0.999, w_B_norm ≈ 0.001
#   fused ≈ 0.999*0.8 + 0.001*0.1 ≈ 0.799 (dominated by A)
res_f2 = fuser.fuse(
    predictions={"A": 0.8, "B": 0.1},
    uncertainties={"A": 0.001, "B": 1.0},
    reliabilities={"A": 1.0, "B": 1.0}
)
# A should dominate heavily
check("Fusion-4.2: low-uncertainty A dominates → fused close to A's pred (0.8)",
      abs(res_f2.probability - 0.8) < 0.05,
      f"got {res_f2.probability:.6f}, expected close to 0.8")

# ── Test 4.3 ──────────────────────────────────────────────────────────────
# By hand: weights sum to 1 always
total_w = sum(res_f2.weights.values())
check("Fusion-4.3: weights always sum to 1.0",
      abs(total_w - 1.0) < 1e-6,
      f"weights sum = {total_w:.8f}, expected 1.0")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DataGenerator — no data leakage
# Three consecutive calls produce three DIFFERENT datasets
# ══════════════════════════════════════════════════════════════════════════════

from cognix.evaluation.scenarios import DataGenerator

print("\n" + "═"*70)
print("SECTION 5: DataGenerator — data independence")
print("═"*70)

gen = DataGenerator(n_samples=100, seed=42)
X_train, y_train = gen.generate_base(4)
X_cal,   y_cal   = gen.generate_base(4)
X_test,  y_test  = gen.generate_base(4)

check("DataGenerator-5.1: train != cal (no leakage)",
      not np.allclose(X_train, X_cal),
      f"train[:2,0]={X_train[:2,0]}, cal[:2,0]={X_cal[:2,0]}")

check("DataGenerator-5.2: cal != test (no leakage)",
      not np.allclose(X_cal, X_test),
      f"cal[:2,0]={X_cal[:2,0]}, test[:2,0]={X_test[:2,0]}")

# ── Test 5.3 ──────────────────────────────────────────────────────────────
# Same seed should reproduce same train split
gen2 = DataGenerator(n_samples=100, seed=42)
X_train2, _ = gen2.generate_base(4)
check("DataGenerator-5.3: same seed reproduces identical train split",
      np.allclose(X_train, X_train2),
      f"max diff: {np.max(np.abs(X_train-X_train2)):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LatencyTracker — percentile math
# ══════════════════════════════════════════════════════════════════════════════

from cognix.metrics.evaluation import LatencyTracker

print("\n" + "═"*70)
print("SECTION 6: LatencyTracker")
print("═"*70)

# By hand:
#   Records = [10, 20, 30, 40, 50] ms
#   p50 = 30.0, p95 = 48.0, mean = 30.0
tracker = LatencyTracker()
for v in [10, 20, 30, 40, 50]:
    tracker.record("test", float(v))
p = tracker.get_percentiles("test")

check("LatencyTracker-6.1: p50 of [10,20,30,40,50] = 30.0",
      abs(p["p50"] - 30.0) < 0.01,
      f"got p50={p['p50']:.2f}, expected 30.0")

check("LatencyTracker-6.2: mean of [10,20,30,40,50] = 30.0",
      abs(p["mean"] - 30.0) < 0.01,
      f"got mean={p['mean']:.2f}, expected 30.0")

check("LatencyTracker-6.3: empty stage returns zeros, no crash",
      tracker.get_percentiles("nonexistent")["p50"] == 0.0,
      "")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"RESULTS: {passed}/{total} passed")
if failed > 0:
    print(f"\nFAILED TESTS ({failed}):")
    for name, ok in results:
        if not ok:
            print(f"  ✗ {name}")
    sys.exit(1)
else:
    print("\n✓ ALL TESTS PASSED — mathematical primitives are correct.")
    sys.exit(0)
