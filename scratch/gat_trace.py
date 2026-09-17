"""
COGNIX GAT Minimal Reproducible Trace.

Shows every intermediate quantity for one agent configuration:
W, Wh, raw attention scores, epistemic prior, masked softmax,
aggregated H_prime, and final activation output.

Does NOT modify the GAT. Reports exact numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cognix.graph.gat import GATLayer
from cognix.graph.epistemic_gat import EpistemicGATLayer

np.random.seed(42)

DIVIDER = "=" * 68
SUB = "-" * 68

# ── Exact inputs from the live audit ─────────────────────────────────
node_features = np.array([
    [0.505222, 0.000149, 0.249818],   # Agent A: [p, epi, ale]
    [0.183543, 0.002564, 0.143949],   # Agent B
    [0.903161, 0.003676, 0.108213],   # Agent C
], dtype=np.float32)

adjacency = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], dtype=np.float32)

# Epistemic prior column j = 1/(1+sigma_e_j)
sigma_e = [0.000149, 0.002564, 0.003676]
W_prior = np.zeros((3, 3))
for j, s in enumerate(sigma_e):
    W_prior[:, j] = 1.0 / (1.0 + s)

epistemic_uncertainties = {
    "Agent_A": 0.000149,
    "Agent_B": 0.002564,
    "Agent_C": 0.003676,
}
agent_order = ["Agent_A", "Agent_B", "Agent_C"]

N, F_in, F_out = 3, 3, 4

print(DIVIDER)
print("COGNIX GAT MINIMAL REPRODUCIBLE TRACE (seed=42)")
print(DIVIDER)

# ── Create one EpistemicGATLayer (same random init as EpistemicGAT) ──
np.random.seed(42)
layer = EpistemicGATLayer(in_features=F_in, out_features=F_out)

print("\n[1] INPUT NODE FEATURES (N=3, F=3)")
print(f"    {node_features}")

print("\n[2] WEIGHT MATRIX W (F_in x F_out)")
print(f"    shape: {layer.W.shape}")
print(f"    W =\n{layer.W.round(4)}")
print(f"    Xavier limit = sqrt(6/(3+4)) = {np.sqrt(6/(F_in+F_out)):.4f}")
print(f"    W range: [{layer.W.min():.4f}, {layer.W.max():.4f}]")

print("\n[3] LINEAR PROJECTION Wh = H @ W  (N x F_out)")
Wh = node_features @ layer.W
print(f"    Wh =\n{Wh.round(4)}")
print(f"    Wh range: [{Wh.min():.4f}, {Wh.max():.4f}]")

print("\n[4] ATTENTION VECTOR a (2*F_out x 1)")
print(f"    shape: {layer.a.shape}")
print(f"    a = {layer.a.T.round(4)}")

print("\n[5] RAW ATTENTION SCORES e_ij = concat(Wh_i, Wh_j) @ a")
a_input = np.zeros((N, N, 2*F_out))
for i in range(N):
    for j in range(N):
        a_input[i, j, :] = np.concatenate([Wh[i], Wh[j]])
e = (a_input @ layer.a).squeeze(-1)
print(f"    Raw e (before LeakyReLU):\n{e.round(4)}")
e_leaky = np.where(e > 0, e, 0.2 * e)
print(f"    After LeakyReLU(alpha=0.2):\n{e_leaky.round(4)}")
print(f"    e_ij range: [{e_leaky.min():.4f}, {e_leaky.max():.4f}]")

print("\n[6] EPISTEMIC PRIOR W_prior[i,j] = 1/(1+sigma_e_j)")
print(f"    (column j = weight for agent j as sender)")
print(f"    {W_prior.round(6)}")
print(f"    Prior range: [{W_prior.min():.6f}, {W_prior.max():.6f}]")
print(f"    Max epistemic contrast = {W_prior.max() - W_prior.min():.6e}")
print("    NOTE: sigma_e values are very small (0.0001-0.004).")
print("    Therefore W_prior values differ by < 0.004 -> near-uniform prior.")
print("    The epistemic prior has MINIMAL effect on attention for this input.")

print("\n[7] EPISTEMICALLY BIASED SCORES e_epistemic = e * W_prior")
e_epistemic = e_leaky * W_prior
print(f"    {e_epistemic.round(6)}")

print("\n[8] MASKED SCORES (non-neighbor -> -inf)")
mask_old = np.where(adjacency > 0, 1, -np.inf)      # ORIGINAL: +1 for edges
mask_new = np.where(adjacency > 0, 0.0, -1e9)       # PROPOSED: +0 for edges
e_masked_old = e_epistemic + mask_old
e_masked_new = e_epistemic + mask_new
print(f"    Original mask (+1 for edges):\n{e_masked_old.round(4)}")
print(f"    Proposed mask (+0 for edges):\n{e_masked_new.round(4)}")
print("    ISSUE: Original mask adds +1 to ALL edges, inflating scores artificially.")
print("    This biases attention away from the actual learned attention scores.")

print("\n[9] SOFTMAX ATTENTION WEIGHTS (original masking)")
e_max_old = np.max(e_masked_old, axis=1, keepdims=True)
exp_old = np.exp(e_masked_old - e_max_old)
attn_old = exp_old / np.sum(exp_old, axis=1, keepdims=True)
print(f"    alpha_ij (original):\n{attn_old.round(4)}")
for i in range(N):
    print(f"    row[{i}] sum = {attn_old[i].sum():.6f}")

print("\n[10] H_prime = activation(alpha @ Wh)")
H_agg = attn_old @ Wh
print(f"    alpha @ Wh =\n{H_agg.round(6)}")
H_relu = np.maximum(0, H_agg)
print(f"    After ReLU:\n{H_relu.round(6)}")
print(f"    ReLU ZEROS: {(H_relu == 0).sum()} out of {H_relu.size} values")
print(f"    H_prime[:, 0] (used for sigmoid) = {H_relu[:, 0]}")
print(f"    sigmoid(H_prime[:, 0]) = {(1/(1+np.exp(-H_relu[:,0]))).round(6)}")

print("\n[11] DIAGNOSIS")
print(SUB)
print("    H_agg range BEFORE ReLU: "
      f"[{H_agg.min():.4f}, {H_agg.max():.4f}]")
if H_agg.min() >= -0.001:
    print("    H_agg is near-zero or positive -> ReLU passes all values")
elif H_agg.max() <= 0.001:
    print("    H_agg is near-zero or negative -> ReLU KILLS all values [COLLAPSE]")
else:
    print("    H_agg has mixed signs -> ReLU causes partial collapse")

all_zero = np.allclose(H_relu[:, 0], 0, atol=1e-4)
print(f"    sigmoid(H_prime[:,0]) all = 0.5? {all_zero}")

print("\n[12] EVALUATE THREE ARCHITECTURES")
print(SUB)

# Option A: Identity on final layer (linear output)
H_identity = H_agg.copy()
p_a = 1/(1+np.exp(-H_identity[:,0]))
print(f"    Option A (Identity final layer):")
print(f"      H_prime[:,0] = {H_identity[:,0].round(6)}")
print(f"      sigmoid = {p_a.round(6)}")
print(f"      Unique probs? {len(np.unique(p_a.round(4))) > 1}")
print(f"      Preserves agent ordering? {bool(p_a[2] > p_a[0] > p_a[1])}")

# Option B: ReLU on hidden, identity on final
np.random.seed(42)
layer1 = EpistemicGATLayer(F_in, 8)
np.random.seed(99)
layer2 = EpistemicGATLayer(8, F_out)
H1 = node_features @ layer1.W
H1_agg = (attn_old @ H1)
H1_relu = np.maximum(0, H1_agg)   # ReLU on hidden layer
H2 = H1_relu @ layer2.W
# For the output layer, approximate attention (simplified)
a2_input = np.zeros((N, N, 2*F_out))
for i in range(N):
    for j in range(N):
        a2_input[i,j,:] = np.concatenate([H2[i], H2[j]])
e2 = (a2_input @ layer2.a).squeeze(-1)
e2l = np.where(e2 > 0, e2, 0.2*e2)
e2ep = e2l * W_prior[:,:F_out] if F_out <= 3 else e2l * W_prior
e2m = e2ep + mask_new
e2max = np.max(e2m, axis=1, keepdims=True)
exp2 = np.exp(e2m - e2max) * (adjacency > 0)
attn2 = exp2 / (exp2.sum(axis=1, keepdims=True) + 1e-9)
H2_agg = attn2 @ H2
p_b = 1/(1+np.exp(-H2_agg[:,0]))
print(f"\n    Option B (ReLU hidden + Identity final):")
print(f"      H_prime[:,0] = {H2_agg[:,0].round(6)}")
print(f"      sigmoid = {p_b.round(6)}")
print(f"      Unique probs? {len(np.unique(p_b.round(4))) > 1}")

# Option C: Trained GAT — requires fitting on training data
print(f"\n    Option C (Trained GAT):")
print(f"      Requires: PyTorch GAT, training on X_train, explicit loss function.")
print(f"      Status: Not yet implemented.")
print(f"      Training target: Collective BCE(p_collective, y_train)")
print(f"      Advantage: Mathematically consistent with 'learned GAT' claim.")
print(f"      Risk: Overfitting on small synthetic dataset.")
print(f"      Required for: Publication-grade 'EpistemicGAT improves accuracy' claim.")

print("\n[13] ARCHITECTURE DECISION")
print(SUB)
print("    COGNIX design intent (from architecture docs):")
print("      M4 = Epistemic Graph Attention")
print("      Scientific claim: epistemic prior on attention (w_j = 1/(1+sigma_e_j))")
print("      NOT a claim about learned feature representations.")
print()
print("    Current RQ_SYNTHETIC_001 setup:")
print("      - GAT weights W, a are randomly initialized (np.random.uniform)")
print("      - They are NOT trained end-to-end")
print("      - Node features are [p_i, sigma_e_i, sigma_a_i] (low-dimensional, ~1)")
print("      - The scientific signal lives in the epistemic attention weighting,")
print("        not in the linear projection Wh")
print()
print("    Verdict:")
print("      Option C is most rigorous but requires a training protocol.")
print("      Option B (ReLU hidden + linear final) is correct intermediate step.")
print("      Option A alone is too minimal.")
print()
print("    RECOMMENDED: Option C — implement a proper PyTorch GAT training step.")
print("      Train on X_train only. Objective: minimize BCE of GAT-fused output.")
print("      Document W as trained, not random.")
print("      Until training is implemented, use Option B with fixed seed and")
print("      document explicitly: 'GAT parameters are random projections, not trained.'")
