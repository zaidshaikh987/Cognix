"""
COGNIX Final Scientific Validation Audit Script.

Runs a complete traced sample through every module, verifies mathematical
correctness, and produces a module-by-module execution table.
"""
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any

# ── Add project root to path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.calibration.conformal import ConformalPredictor, evaluate_coverage
from cognix.explainability.epistemic_shapley import EpistemicShapley
from cognix.belief.fusion import CognixBeliefFuser
from cognix.belief.base import BeliefState, FusionStrategy
from cognix.engine.pipeline import CognixPipeline
from cognix.config.schema import CognixConfig

np.random.seed(42)
torch.manual_seed(42)

DIVIDER = "=" * 72
SUB = "-" * 72

# ── Minimal agent for isolated tracing ───────────────────────────────
class AuditAgent:
    """Exact same architecture as HeterogeneousAgent in rq_synthetic_001.py."""
    def __init__(self, name, feature_idx, T=30):
        self.agent_id = name
        self.name = name
        self.feature_idx = feature_idx
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.healthy = True

    def fit(self, X, y, epochs=50):
        opt = torch.optim.Adam(self.net.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        self.net.train()
        Xt = torch.FloatTensor(X)
        yt = torch.FloatTensor(y).view(-1, 1)
        for _ in range(epochs):
            opt.zero_grad()
            o = self.net(Xt[:, self.feature_idx:self.feature_idx+1])
            loss_fn(o, yt).backward()
            opt.step()

    def predict(self, x):
        if not self.healthy: return 0.5
        self.net.eval()
        with torch.no_grad():
            feat = torch.FloatTensor(x).view(1, -1)
            return float(self.net(feat[:, self.feature_idx:self.feature_idx+1]).item())

    def estimate_uncertainty(self, x):
        if not self.healthy:
            class UQ: epistemic=1.0; aleatoric=1.0; total=2.0
            return UQ()
        self.net.train()  # enable dropout
        with torch.no_grad():
            feat = torch.FloatTensor(x).view(1, -1)
            preds = np.array([
                self.net(feat[:, self.feature_idx:self.feature_idx+1]).item()
                for _ in range(self.T)
            ])
        p_bar = np.mean(preds)
        epi  = float(np.mean((preds - p_bar)**2))
        ale  = float(np.mean(preds * (1 - preds)))
        class UQ: pass
        uq = UQ()
        uq.epistemic = epi
        uq.aleatoric = ale
        uq.total = epi + ale
        uq.preds = preds  # expose raw samples for audit
        return uq


# ── Data generation ───────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 300
X = rng.normal(0, 1, (n, 3))
logits = 1.5*X[:,0] - 2.0*X[:,1] + 0.5*X[:,2]**2
probs = 1/(1+np.exp(-logits))
y = (rng.random(n) < probs).astype(float)

n_train = int(0.6*n)
n_cal = int(0.2*n)
X_train, y_train = X[:n_train], y[:n_train]
X_cal,   y_cal   = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
X_test,  y_test  = X[n_train+n_cal:], y[n_train+n_cal:]

# ── Train agents ─────────────────────────────────────────────────────
agents = [
    AuditAgent("Agent_A_Feat0", 0),
    AuditAgent("Agent_B_Feat1", 1),
    AuditAgent("Agent_C_Feat2", 2),
]
for ag in agents:
    ag.fit(X_train, y_train, epochs=30)

# ── Pick sample 0 from test set for trace ────────────────────────────
x0 = X_test[0]
y0 = y_test[0]

print(DIVIDER)
print("COGNIX FINAL SCIENTIFIC VALIDATION AUDIT")
print(DIVIDER)


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: Agent Predictions
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 1] AGENT PREDICTIONS")
print(SUB)
print(f"  Input x0 = {x0}")
print(f"  Ground truth y0 = {y0}")

predictions = {}
for ag in agents:
    p = ag.predict(x0)
    predictions[ag.agent_id] = p
    print(f"  {ag.agent_id}: predict() = {p:.6f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: MC Dropout Uncertainty Quantification (M1: UDE)
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 2] MC DROPOUT UNCERTAINTY QUANTIFICATION (T=30)")
print(SUB)
uncertainties = {}
for ag in agents:
    uq = ag.estimate_uncertainty(x0)
    uncertainties[ag.agent_id] = {
        "epistemic": uq.epistemic,
        "aleatoric": uq.aleatoric,
        "total": uq.total,
    }
    p_bar = np.mean(uq.preds)
    epi_check = float(np.mean((uq.preds - p_bar)**2))
    ale_check = float(np.mean(uq.preds * (1 - uq.preds)))
    print(f"\n  {ag.agent_id}:")
    print(f"    T={ag.T} MC passes | dropout active = {ag.net.training}")
    print(f"    p_bar (mean_p)     = {p_bar:.6f}")
    print(f"    epistemic = Var(p_t)        = {epi_check:.6f}  [reported: {uq.epistemic:.6f}]")
    print(f"    aleatoric = E[p_t(1-p_t)]   = {ale_check:.6f}  [reported: {uq.aleatoric:.6f}]")
    print(f"    total     = epi + ale        = {(epi_check+ale_check):.6f}  [reported: {uq.total:.6f}]")
    print(f"    Consistency check: |Var+E[]-total| = {abs(uq.total - uq.epistemic - uq.aleatoric):.2e}")
    print(f"    p_t std across passes        = {np.std(uq.preds):.6f}")
    print(f"    HARDCODED aleatoric? {abs(uq.aleatoric - 0.05) < 1e-6}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: Trust Weights
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 3] EPISTEMIC TRUST WEIGHTS")
print(SUB)
eps = 1e-8
raw_w = {a: 1.0 / (uncertainties[a]["epistemic"] + eps) for a in uncertainties}
total_w = sum(raw_w.values())
trust_weights = {a: raw_w[a] / total_w for a in raw_w}
for a, w in trust_weights.items():
    print(f"  {a}: w = 1/(sigma_e + eps) normalized = {w:.6f}  (sigma_e={uncertainties[a]['epistemic']:.6f})")
print(f"  Sum of weights = {sum(trust_weights.values()):.8f} (must = 1.0)")


# ══════════════════════════════════════════════════════════════════════
# STAGE 4: EpistemicGAT (M4)
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 4] EPISTEMIC GAT FORWARD PASS (M4)")
print(SUB)

gat = EpistemicGAT(num_layers=2, input_dim=3, hidden_dim=8, output_dim=4)
agent_order = list(predictions.keys())
N = len(agent_order)

node_features = np.array([
    [predictions[a], uncertainties[a]["epistemic"], uncertainties[a]["aleatoric"]]
    for a in agent_order
], dtype=np.float32)
adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
epi_unc = {a: uncertainties[a]["epistemic"] for a in agent_order}

print(f"  node_features shape  = {node_features.shape}  (N={N}, F=3: [p_i, sigma_e, sigma_a])")
print(f"  adjacency shape      = {adjacency.shape}  (fully connected, no self-loops)")
print(f"  agent_order          = {agent_order}")
print(f"  epistemic_prior = 1/(1+sigma_e):")

W_prior = gat.compute_epistemic_weights(epi_unc, agent_order)
for j, a in enumerate(agent_order):
    print(f"    w_initial(j={j},{a}) = {W_prior[0,j]:.6f}")

H_prime, attn_list = gat.forward(node_features, adjacency, epi_unc, agent_order)

print(f"\n  H_prime shape = {H_prime.shape}  (N x out_dim)")
print(f"  Attention layers: {len(attn_list)}")
attn = attn_list[-1]
print(f"  Attention matrix (last layer):")
for i in range(N):
    row_sum = attn[i].sum()
    print(f"    row[{i}] = {attn[i].round(4)}  sum={row_sum:.6f}")

print("\n  Probability extraction: p_i_refined = sigmoid(H_prime[i, 0])")
refined_probs = 1.0 / (1.0 + np.exp(-H_prime[:, 0]))
refined_predictions = {}
for idx, a in enumerate(agent_order):
    refined_predictions[a] = float(np.clip(refined_probs[idx], 1e-7, 1-1e-7))
    print(f"  {a}: H_prime[{idx},0]={H_prime[idx,0]:.6f}  -> sigmoid={refined_predictions[a]:.6f}")
    print(f"    (original prediction was {predictions[a]:.6f})")


# ══════════════════════════════════════════════════════════════════════
# STAGE 5: Bayesian Belief Fusion (M2)
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 5] BAYESIAN BELIEF FUSION (M2: epistemic_weighted)")
print(SUB)

fuser = CognixBeliefFuser()
fuser.default_strategy = FusionStrategy.EPISTEMIC_WEIGHTED

beliefs = []
for a in agent_order:
    p = refined_predictions[a]
    beliefs.append(BeliefState(
        agent_id=a,
        belief=np.array([1-p, p]),
        alpha=p*10, beta_param=(1-p)*10,
        confidence=p,
    ))
    print(f"  {a}: belief=[{1-p:.4f}, {p:.4f}]  confidence={p:.4f}")

reliabilities_audit = {ag.agent_id: 1.0 for ag in agents}  # Fixed at 1.0 — documented
print(f"\n  Reliabilities: {reliabilities_audit}")
print("  NOTE: Reliability = 1.0 for all agents (intentional experimental control)")
print("  Fusion weights = reliability_j / (sigma_e_j + eps), then normalized")

fused = fuser.fuse(
    beliefs=beliefs,
    strategy=FusionStrategy.EPISTEMIC_WEIGHTED,
    epistemic_uncertainties={a: uncertainties[a]["epistemic"] for a in agent_order},
    reliabilities=reliabilities_audit,
)
print(f"\n  Fused belief = {fused.belief}")
print(f"  Fused confidence (p_collective) = {fused.confidence:.6f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 6: Conformal Calibration (M3)
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 6] CONFORMAL CALIBRATION (M3)")
print(SUB)

# Run M1->M4->M2 on ALL calibration samples (same pipeline as test)
print(f"  Running pre-calibration pipeline on {len(X_cal)} calibration samples...")

def run_collective(x_i):
    preds_i = {ag.agent_id: ag.predict(x_i) for ag in agents}
    uncs_i = {}
    for ag in agents:
        uq = ag.estimate_uncertainty(x_i)
        uncs_i[ag.agent_id] = {"epistemic": uq.epistemic, "aleatoric": uq.aleatoric}
    # trust weights
    rw = {a: 1.0/(uncs_i[a]["epistemic"]+1e-8) for a in preds_i}
    tw = {a: rw[a]/sum(rw.values()) for a in rw}
    # GAT
    ao = list(preds_i.keys())
    nf = np.array([[preds_i[a], uncs_i[a]["epistemic"], uncs_i[a]["aleatoric"]] for a in ao], dtype=np.float32)
    adj = (np.ones((len(ao), len(ao))) - np.eye(len(ao))).astype(np.float32)
    eu = {a: uncs_i[a]["epistemic"] for a in ao}
    Hp, _ = gat.forward(nf, adj, eu, ao)
    rp = 1.0/(1.0+np.exp(-Hp[:,0]))
    for idx, a in enumerate(ao):
        preds_i[a] = float(np.clip(rp[idx], 1e-7, 1-1e-7))
    # fusion
    bls = [BeliefState(a, np.array([1-preds_i[a], preds_i[a]]), preds_i[a]*10, (1-preds_i[a])*10, preds_i[a]) for a in ao]
    fd = fuser.fuse(bls, FusionStrategy.EPISTEMIC_WEIGHTED,
                    epistemic_uncertainties={a: uncs_i[a]["epistemic"] for a in ao},
                    reliabilities={a: 1.0 for a in ao})
    return float(np.clip(fd.confidence, 1e-7, 1-1e-7))

cal_probs = []
for i in range(len(X_cal)):
    pc = run_collective(X_cal[i])
    cal_probs.append([1-pc, pc])
cal_probs = np.array(cal_probs)

cp = ConformalPredictor()
cp.calibrate(cal_probs, y_cal.astype(int))
print(f"  n_cal = {cp.n_cal}")
print(f"  cal_scores (nonconformity, sorted) = {cp.cal_scores.round(4)}")
q_idx = int(np.ceil((cp.n_cal + 1) * (1 - 0.05)))
q_val = cp.cal_scores[q_idx-1] if q_idx <= cp.n_cal else 1.0
print(f"  Quantile index q_idx = ceil({cp.n_cal+1} * 0.95) = {q_idx}")
print(f"  Quantile q_hat = {q_val:.6f}")

# Run on test sample x0
p0_collective = run_collective(x0)
test_input = np.array([[1-p0_collective, p0_collective]])
pred_sets = cp.predict(test_input, alpha=0.05)
ps0 = pred_sets[0]
print(f"\n  Test sample x0: p_collective = {p0_collective:.6f}")
print(f"  Nonconformity scores for x0: {(1.0 - test_input[0]).round(4)}")
print(f"  Prediction set (alpha=0.05): {ps0.prediction_set}")
print(f"  Target coverage: {ps0.coverage_target}")
print(f"  Quantile used: {ps0.quantile:.6f}")
print(f"  True label y0={y0:.0f}  in prediction_set? {int(y0) in ps0.prediction_set}")

# Coverage on full test set
test_probs_all = []
for i in range(len(X_test)):
    pc = run_collective(X_test[i])
    test_probs_all.append([1-pc, pc])
test_probs_all = np.array(test_probs_all)
all_pred_sets = cp.predict(test_probs_all, alpha=0.05)
coverage = evaluate_coverage(all_pred_sets, y_test.astype(int))
avg_set_size = np.mean([len(ps.prediction_set) for ps in all_pred_sets])
print(f"\n  Empirical coverage on test set (n={len(X_test)}): {coverage:.4f}")
print(f"  Target coverage: 0.95")
print(f"  Coverage >= 0.90? {'YES' if coverage >= 0.90 else 'NO — BELOW TARGET'}")
print(f"  Average prediction set size: {avg_set_size:.4f}")
print(f"  WARNING: n_cal={cp.n_cal} is small. Finite-sample guarantee may not hold.")


# ══════════════════════════════════════════════════════════════════════
# STAGE 7: Epistemic Shapley Attribution (M5)
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 7] EPISTEMIC SHAPLEY ATTRIBUTION (M5)")
print(SUB)

def v_collective(subset):
    """v(S): collective fused confidence using only agents in S."""
    if not subset:
        return 1.0
    sub_preds = {a: refined_predictions[a] for a in subset}
    sub_uncs = {a: uncertainties[a] for a in subset}
    sub_bls = [BeliefState(a, np.array([1-sub_preds[a], sub_preds[a]]),
                           sub_preds[a]*10, (1-sub_preds[a])*10, sub_preds[a])
               for a in subset]
    fd = fuser.fuse(sub_bls, FusionStrategy.EPISTEMIC_WEIGHTED,
                    epistemic_uncertainties={a: sub_uncs[a]["epistemic"] for a in subset},
                    reliabilities={a: 1.0 for a in subset})
    return float(fd.confidence)

es = EpistemicShapley()
phi = es.compute(agent_order, v_collective, num_samples=500)

v_all = v_collective(agent_order)
v_empty = v_collective([])
total_phi = sum(phi.values())
efficiency_error = abs(total_phi - (v_all - v_empty))

print(f"  v(all agents) = {v_all:.6f}")
print(f"  v(empty set)  = {v_empty:.6f}")
print(f"  v(N) - v(∅)   = {v_all - v_empty:.6f}")
print(f"  sum(phi_i)    = {total_phi:.6f}")
print(f"  Efficiency property |sum(phi) - (v(N)-v(∅))| = {efficiency_error:.2e}")
print(f"  Efficiency satisfied (< 0.05)? {'YES' if efficiency_error < 0.05 else 'NO'}")
for a, pv in phi.items():
    print(f"  phi({a}) = {pv:.6f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 8: Hidden Fallback Scan
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 8] HIDDEN FALLBACK SCAN IN RESEARCH MODE")
print(SUB)

pipe = CognixPipeline(
    belief_fuser=fuser,
    graph=gat,
    calibrator=cp,
    attribution=es,
    mode="research",
)
result = pipe.run(agents, x0, {})
ms = result.metadata.get("module_status", {})
print("  Module execution status from live research run:")
for mod, status in ms.items():
    executed = status.get("executed", False)
    method = status.get("method", "?")
    dur = status.get("duration_ms", 0)
    reason = status.get("failure_reason", None)
    flag = "✓" if executed else "✗ FAILED"
    print(f"  [{flag}] {mod:15s}: method={method}, duration={dur:.3f}ms, failure={reason}")

print(f"\n  Calibration info from trace: {result.calibration_metrics}")
print(f"  Communication from trace: keys={list((result.communication_statistics or {}).keys())}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 9: Degradation Sensitivity Test
# ══════════════════════════════════════════════════════════════════════
print("\n[STAGE 9] DEGRADATION SENSITIVITY TEST")
print(SUB)

# OOD sample: feature 1 shifted by +10
x_ood = x0.copy()
x_ood[1] += 10.0
print(f"  Normal sample:  x[1]={x0[1]:.4f}")
print(f"  OOD sample:     x[1]={x_ood[1]:.4f}")

epi_normal, ale_normal = [], []
epi_ood, ale_ood = [], []

for ag in agents:
    uq_n = ag.estimate_uncertainty(x0)
    uq_o = ag.estimate_uncertainty(x_ood)
    epi_normal.append(uq_n.epistemic)
    ale_normal.append(uq_n.aleatoric)
    epi_ood.append(uq_o.epistemic)
    ale_ood.append(uq_o.aleatoric)
    print(f"  {ag.agent_id}: normal epi={uq_n.epistemic:.4f}, ale={uq_n.aleatoric:.4f}"
          f"  |  ood epi={uq_o.epistemic:.4f}, ale={uq_o.aleatoric:.4f}")

print(f"\n  Mean epistemic: normal={np.mean(epi_normal):.4f}  ood={np.mean(epi_ood):.4f}")
print(f"  Mean aleatoric: normal={np.mean(ale_normal):.4f}  ood={np.mean(ale_ood):.4f}")
print(f"  Epistemic increases under OOD? {'YES' if np.mean(epi_ood) > np.mean(epi_normal) else 'NO — CHECK'}")

print(f"\n{DIVIDER}")
print("AUDIT COMPLETE — SEE audit_results.json FOR FULL NUMBERS")
print(DIVIDER)

results = {
    "sample": {
        "x0": x0.tolist(), "y0": float(y0),
        "predictions": predictions,
        "uncertainties": uncertainties,
        "trust_weights": trust_weights,
        "refined_predictions": refined_predictions,
        "fused_confidence": float(fused.confidence),
        "prediction_set": ps0.prediction_set,
        "quantile": float(q_val),
        "n_cal": int(cp.n_cal),
        "coverage_test": float(coverage),
        "avg_set_size": float(avg_set_size),
        "shapley": phi,
        "efficiency_error": float(efficiency_error),
    },
    "module_status": ms,
}
with open("results/audit_results.json", "w") as f:
    json.dump(results, f, indent=2)
