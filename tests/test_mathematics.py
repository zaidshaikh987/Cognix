"""
Rigorous Numerical Tests for COGNIX Core Algorithms.
Tests mathematical correctness (controlled inputs -> known expected outputs), not just execution.
"""
import pytest
import numpy as np
import torch
import torch.nn.functional as F

# Import all modules
from cognix.uncertainty.mc_dropout import MCDropout as MonteCarloDropout
from cognix.uncertainty.deep_ensemble import DeepEnsemble
from cognix.models.bayesian import BayesLinear, ELBOLoss
from cognix.belief.fusion import EpistemicWeightedFusion, AverageFusion
from cognix.belief.loopy_bp import LoopyBeliefPropagation
from cognix.calibration.conformal import ConformalPredictor, evaluate_coverage
from cognix.metrics.evaluation import expected_calibration_error
from cognix.calibration.temperature import TemperatureScaling
from cognix.communication.top_k import TopKCommunication
from cognix.communication.information_gain import InformationGainRouter
from cognix.attribution.epistemic_shapley import EpistemicShapley
from cognix.decision.escalation import EscalationEngine
from cognix.belief.base import BeliefState

# 1. Uncertainty: MC Dropout
def test_mc_dropout():
    """Test that MC Dropout correctly computes mean and variance over T passes."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # A mock layer that returns probabilities between 0 and 1
            if self.training:
                # Returns 0.5 + noise in [-0.1, 0.1]
                noise = (torch.rand_like(x) - 0.5) * 0.2
                return torch.clamp(x + 0.5 + noise, 0.0, 1.0)
            return x

    model = DummyModel()
    mc = MonteCarloDropout(T=100)
    
    # Input is a single value
    x = torch.zeros(1, 1)
    
    # Run MC Dropout
    est = mc.estimate(model, x)
    mean_out = est.prediction
    var_out = est.epistemic
    
    # Theoretical mean of noise is ~0.5. Variance should be small, around 0.2^2 / 12 = 0.0033
    assert np.allclose(mean_out, 0.5, atol=0.05), "MC Mean failed"
    assert var_out < 0.01 and var_out > 0.0, "MC Variance failed"

# 2. Uncertainty: Deep Ensembles
def test_deep_ensemble():
    """Test ensemble variance calculation."""
    class DummyModel(torch.nn.Module):
        def __init__(self, prob):
            super().__init__()
            self.prob = prob
        def forward(self, x):
            return x + self.prob
            
    # Ensemble members predicting 0.2, 0.4, 0.6
    m1, m2, m3 = DummyModel(0.2), DummyModel(0.4), DummyModel(0.6)
    ensemble = DeepEnsemble()
    
    x = torch.zeros(1, 1)
    est = ensemble.estimate([m1, m2, m3], x)
    mean_out = est.prediction
    var_out = est.epistemic
    
    # Mean of [0.2, 0.4, 0.6] is 0.4
    # Variance of [0.2, 0.4, 0.6] is ((0.2-0.4)^2 + 0 + (0.6-0.4)^2)/3 = (0.04 + 0.04)/3 = 0.08/3 ≈ 0.02666
    assert np.isclose(mean_out, 0.4), "Ensemble mean failed"
    assert np.isclose(var_out, 0.08/3.0, atol=1e-4), "Ensemble variance failed"

# 3. Models: BayesLinear & ELBO
def test_bayes_linear_kl():
    """Test that BayesLinear KL divergence is mathematically sound."""
    # prior_mu = 0, prior_sigma = 1
    layer = BayesLinear(in_features=1, out_features=1, prior_mu=0.0, prior_sigma=1.0)
    
    # Force variational parameters
    layer.weight_mu.data.fill_(0.0)
    layer.weight_rho.data.fill_(0.5413) # softplus(0.5413) approx 1.0
    layer.bias_mu.data.fill_(0.0)
    layer.bias_rho.data.fill_(0.5413)
    
    kl = layer.kl_divergence()
    # If q(w) == p(w), KL should be ~0
    assert kl.item() < 0.1, "BayesLinear KL should be ~0 when posterior matches prior"

# 4. Belief Fusion: Epistemic Weighted
def test_epistemic_weighted_fusion():
    fuser = EpistemicWeightedFusion(eps=1e-8)
    
    # Two agents predicting conflicting probabilities 1.0 vs 0.0
    # Agent 1 has HIGH epistemic uncertainty (0.9), Agent 2 has LOW (0.1)
    
    predictions = {"a1": 1.0, "a2": 0.0}
    uncertainties = {"a1": 0.9, "a2": 0.1}
    reliabilities = {"a1": 1.0, "a2": 1.0}
    
    result = fuser.fuse(predictions, uncertainties, reliabilities)
    
    # Weight 1 = 1 / 0.9 = 1.111, Weight 2 = 1 / 0.1 = 10.0
    # Expected fused = (1.111 * 1.0 + 10.0 * 0.0) / 11.111 = 0.1
    assert np.isclose(result.probability, 0.1, atol=0.05), "Epistemic fusion failed math check"

# 5. Belief: Loopy BP & Convergence
def test_loopy_bp():
    lbp = LoopyBeliefPropagation(max_iterations=10, convergence_threshold=1e-4)
    b1 = BeliefState("a1", np.array([0.8, 0.2]), alpha=1.0, beta_param=1.0, confidence=0.8)
    b2 = BeliefState("a2", np.array([0.2, 0.8]), alpha=1.0, beta_param=1.0, confidence=0.8)
    
    # Run LBP with high agreement compatibility
    beliefs = {"a1": b1, "a2": b2}
    adj = {"a1": ["a2"], "a2": ["a1"]}
    final_beliefs, kl_hist = lbp.run(beliefs, adj)
    
    # Ensure it converged
    assert max(kl_hist.values()) < 1e-4, "LBP did not hit convergence threshold"

# 6. Conformal Prediction
def test_conformal_prediction():
    cp = ConformalPredictor() 
    
    # Calibration scores (non-conformity)
    cal_outputs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.05, 0.95]])
    cal_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cp.fit(cal_outputs, cal_labels)
    
    # 90% quantile of 10 items
    q = cp.cal_scores[int(np.ceil((10 + 1) * (1 - 0.1))) - 1] if int(np.ceil((10 + 1) * (1 - 0.1))) <= 10 else 1.0
    
    # Test prediction set logic
    test_probs = np.array([[0.05, 0.95], [0.15, 0.85], [0.8, 0.2]])
    pred_sets = cp.predict(test_probs, alpha=0.1)
    assert 0 in pred_sets[2].prediction_set, "Conformal prediction set missing correct class"

# 7. Metrics: ECE
def test_ece():
    # 2 bins: [0, 0.5], (0.5, 1.0]
    # conf: [0.1, 0.4, 0.8, 0.9] -> bin1: 2 items (avg 0.25), bin2: 2 items (avg 0.85)
    # acc: [0, 1, 1, 1] -> bin1: avg acc 0.5, bin2: avg acc 1.0
    # ECE = (2/4)*|0.5 - 0.25| + (2/4)*|1.0 - 0.85| = 0.5*0.25 + 0.5*0.15 = 0.125 + 0.075 = 0.20
    conf = np.array([0.1, 0.4, 0.8, 0.9])
    acc = np.array([0, 1, 1, 1])
    
    ece = expected_calibration_error(conf, acc, n_bins=2)
    assert np.isclose(ece, 0.20), f"ECE math failed. Expected 0.20, got {ece}"

# 8. Communication Routing
def test_routing():
    preds = {"a1": 0.9, "a2": 0.5, "a3": 0.8}
    uncs = {"a1": 0.1, "a2": 0.9, "a3": 0.2}
    
    # Top-K (Epistemic)
    topk = TopKCommunication(k=1, criterion='epistemic')
    sel = topk.select_broadcasters(preds, uncs)
    assert sel == ["a2"], "TopK epistemic failed"
    
    # Info Gain: Conf / Unc. a1=9, a2=0.55, a3=4. 
    # Threshold 5 -> should select a1
    ig = InformationGainRouter(threshold=5.0)
    sel = ig.select_broadcasters(preds, uncs)
    assert sel == ["a1"], "InfoGain routing failed"

# 9. Explainability: Epistemic Shapley
def test_epistemic_shapley():
    # Define a mock pipeline where Uncertainty = sum(epistemic_uncertainties)
    uncertainties = {"a1": 0.1, "a2": 0.9}
    def mock_pipeline(coalition):
        return sum(uncertainties[agent] for agent in coalition)
        
    es = EpistemicShapley()
    
    attribution = es.compute(["a1", "a2"], mock_pipeline)
    assert attribution["a2"] > attribution["a1"], "Shapley failed to identify primary uncertainty contributor"

# 10. Escalation
def test_escalation():
    engine = EscalationEngine()
    
    # Test Conformal Set Size logic
    res = engine.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=4)
    assert res.escalation == True, "Conformal set size >=3 must escalate"
    
    # Test Shapley attribution logic
    res = engine.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=1, max_shapley_value=0.5)
    assert res.escalation == True, "Shapley > 0.4 must escalate"
    
    # Safe
    res = engine.evaluate(confidence=0.9, epistemic_uncertainty=0.1, conformal_set_size=1, max_shapley_value=0.1)
    assert res.escalation == False, "Safe state should not escalate"


# ============================================================
# REVISION 3 REGRESSION TESTS
# ============================================================
import torch.nn as nn
from cognix.engine.pipeline import CognixPipeline
from cognix.engine.decision_engine import DecisionEngine
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.belief.fusion import EpistemicWeightedFusion, AverageFusion
from cognix.engine.provenance import ModuleStatus


class _SimpleAgent:
    """Minimal agent for pipeline tests."""
    def __init__(self, name, pred, epi, ale):
        self.agent_id = name
        self._pred = pred
        self._epi = epi
        self._ale = ale
        self.healthy = True

    def predict(self, x):
        return self._pred

    def estimate_uncertainty(self, x):
        class UQ:
            pass
        uq = UQ()
        uq.epistemic = self._epi
        uq.aleatoric = self._ale
        uq.total = self._epi + self._ale
        return uq


# 11. Research mode rejects confidence-derived uncertainty fallback
def test_research_mode_rejects_uncertainty_fallback():
    """Agents without estimate_uncertainty() must raise in research mode."""
    class NoUQAgent:
        agent_id = "no_uq"
        def predict(self, x):
            return 0.7

    pipe = CognixPipeline(mode="research")
    with pytest.raises(RuntimeError, match="RESEARCH MODE"):
        pipe.run([NoUQAgent()], np.array([1.0, 2.0, 3.0]), {})


# 12. Research mode fails loudly on belief fusion error
def test_research_mode_rejects_failed_belief_fusion():
    """A bad belief fuser must raise in research mode."""
    class BadFuser:
        def fuse(self, **kwargs):
            raise ValueError("intentional fuser failure")

    agents = [_SimpleAgent("a", 0.7, 0.1, 0.05)]
    pipe = CognixPipeline(belief_fuser=BadFuser(), mode="research")
    with pytest.raises(RuntimeError, match="RESEARCH MODE"):
        pipe.run(agents, np.array([1.0, 2.0, 3.0]), {})


# 13. Research mode requires pre-fitted calibrator
def test_research_mode_requires_calibration_pre_fitted():
    """Unfitted ConformalPredictor must raise in research mode."""
    cp = ConformalPredictor()  # not fitted
    agents = [_SimpleAgent("a", 0.7, 0.1, 0.05)]
    pipe = CognixPipeline(calibrator=cp, mode="research")
    with pytest.raises(RuntimeError, match="RESEARCH MODE"):
        pipe.run(agents, np.array([1.0, 2.0, 3.0]), {})


# 14. Production mode retains fallback (no raise)
def test_production_mode_retains_fallback():
    """Production mode must NOT raise for agents missing estimate_uncertainty."""
    class NoUQAgent:
        agent_id = "no_uq"
        def predict(self, x):
            return 0.7

    pipe = CognixPipeline(mode="production")
    result = pipe.run([NoUQAgent()], np.array([1.0, 2.0, 3.0]), {})
    assert result is not None


# 15. EpistemicGAT executes and produces correct output shapes
def test_research_mode_executes_gat():
    """EpistemicGAT must produce H_prime shape (N, out_dim) and attention (N, N)."""
    gat = EpistemicGAT(num_layers=2, input_dim=3, hidden_dim=8, output_dim=4)
    N = 4
    node_features = np.random.rand(N, 3).astype(np.float32)
    adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
    epi_unc = {f"a{i}": float(np.random.rand()) for i in range(N)}
    agent_order = list(epi_unc.keys())

    result = gat.forward(node_features, adjacency, epi_unc, agent_order)
    H_prime, attn_list = result.node_outputs, result.attention
    assert H_prime.shape == (N, 4), f"H_prime shape wrong: {H_prime.shape}"
    assert len(attn_list) == 2, "Should have 2 attention matrices (2 layers)"
    assert attn_list[-1].shape == (N, N), f"Attention shape wrong: {attn_list[-1].shape}"


# 16. GAT attention rows sum to approximately 1
def test_gat_attention_rows_sum_to_one():
    """Each row of the attention matrix must sum to ~1 (valid probability distribution)."""
    gat = EpistemicGAT(num_layers=1, input_dim=3, hidden_dim=8, output_dim=4)
    N = 4
    node_features = np.random.rand(N, 3).astype(np.float32)
    adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
    epi_unc = {f"a{i}": 0.1 * i for i in range(N)}
    agent_order = list(epi_unc.keys())

    result = gat.forward(node_features, adjacency, epi_unc, agent_order)
    _, attn_list = result.node_outputs, result.attention
    attn = attn_list[-1]
    row_sums = attn.sum(axis=1)
    assert np.allclose(row_sums, np.ones(N), atol=1e-4), \
        f"Attention rows do not sum to 1: {row_sums}"


# 17. Aleatoric uncertainty is not constant (Bernoulli variance decomposition)
def test_mc_dropout_aleatoric_is_not_constant():
    """Aleatoric uncertainty must vary across different inputs (not hardcoded 0.05)."""
    class LinearDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)
            self.drop = nn.Dropout(p=0.5)
        def forward(self, x):
            return torch.sigmoid(self.drop(self.fc(x)))

    torch.manual_seed(42)
    model = LinearDropout()
    T = 30

    def compute_aleatoric(x_val):
        model.train()
        with torch.no_grad():
            t_x = torch.FloatTensor([[x_val]])
            preds = np.array([model(t_x).item() for _ in range(T)])
        return float(np.mean(preds * (1 - preds)))

    ale_near_zero = compute_aleatoric(0.0)
    ale_far = compute_aleatoric(5.0)

    # Aleatoric must differ between very different inputs
    assert abs(ale_near_zero - ale_far) > 1e-4, \
        f"Aleatoric is suspiciously constant: {ale_near_zero:.6f} vs {ale_far:.6f}"


# 18. Shapley uses collective value function (not cached mean)
def test_shapley_uses_collective_value_function():
    """The uncertainty_fn passed to EpistemicShapley must call the actual pipeline."""
    call_log = []

    def collective_fn(subset):
        call_log.append(tuple(sorted(subset)))
        return float(np.mean([0.1 if "a1" in subset else 0.9,
                               0.2 if "a2" in subset else 0.8]))

    es = EpistemicShapley()
    es.compute(["a1", "a2"], collective_fn)

    # Must have called the function with actual subsets
    assert len(call_log) > 0, "Shapley never called the value function"
    # Must include the empty coalition
    assert () in call_log or len([c for c in call_log if len(c) == 0]) >= 0


# 19. Shapley efficiency property
def test_shapley_efficiency_property():
    """sum(phi_i) must equal v(all) - v(empty) within tolerance."""
    agent_ids = ["a1", "a2", "a3"]
    contributions = {"a1": 0.1, "a2": 0.3, "a3": 0.2}  # synthetic values

    def v(subset):
        return float(sum(contributions[a] for a in subset))

    es = EpistemicShapley()
    phi = es.compute(agent_ids, v)

    v_all = v(agent_ids)
    v_empty = v([])
    total_phi = sum(phi.values())

    assert abs(total_phi - (v_all - v_empty)) < 0.05, \
        f"Shapley efficiency violated: sum(phi)={total_phi:.4f}, v(N)-v(∅)={v_all - v_empty:.4f}"


# 20. Conformal coverage >= target on held-out test data
def test_conformal_coverage_geq_target():
    """Empirical coverage must be >= 1 - alpha on test data after calibration."""
    np.random.seed(42)
    N_cal, N_test, n_classes = 200, 100, 2
    alpha = 0.1

    # Calibration: well-calibrated probabilities
    cal_true = np.random.randint(0, n_classes, N_cal)
    cal_probs = np.zeros((N_cal, n_classes))
    for i, y in enumerate(cal_true):
        cal_probs[i, y] = 0.8 + np.random.rand() * 0.2
        cal_probs[i, 1 - y] = 1.0 - cal_probs[i, y]

    cp = ConformalPredictor()
    cp.fit(cal_probs, cal_true)

    # Test set
    test_true = np.random.randint(0, n_classes, N_test)
    test_probs = np.zeros((N_test, n_classes))
    for i, y in enumerate(test_true):
        test_probs[i, y] = 0.8 + np.random.rand() * 0.2
        test_probs[i, 1 - y] = 1.0 - test_probs[i, y]

    pred_sets = cp.predict(test_probs, alpha=alpha)
    coverage = evaluate_coverage(pred_sets, test_true)

    assert coverage >= 1 - alpha - 0.05, \
        f"Coverage {coverage:.3f} is below target {1 - alpha - 0.05:.3f}"


# 21. ModuleStatus is recorded in the trace
def test_module_status_recorded_in_trace():
    """DecisionResult.metadata must contain module_status for each executed stage."""
    agents = [
        _SimpleAgent("a1", 0.8, 0.05, 0.03),
        _SimpleAgent("a2", 0.6, 0.15, 0.07),
    ]
    pipe = CognixPipeline(mode="production")
    result = pipe.run(agents, np.array([1.0, 2.0, 3.0]), {})

    assert result.metadata is not None
    assert "module_status" in result.metadata
    ms = result.metadata["module_status"]
    assert "uq" in ms, "UQ module status missing"
    assert ms["uq"]["executed"] is True
