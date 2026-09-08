"""
Rigorous Numerical Tests for COGNIX Core Algorithms.
Tests mathematical correctness (controlled inputs -> known expected outputs), not just execution.
"""
import pytest
import numpy as np
import torch
import torch.nn.functional as F

# Import all modules
from cognix.uncertainty.mc_dropout import MonteCarloDropout
from cognix.uncertainty.deep_ensemble import DeepEnsemble
from cognix.models.bayesian import BayesLinear, ELBOLoss
from cognix.belief.fusion import CognixBeliefFuser, FusionStrategy
from cognix.belief.loopy_bp import LoopyBeliefPropagation
from cognix.calibration.conformal import ConformalPredictor
from cognix.metrics.evaluation import expected_calibration_error
from cognix.calibration.temperature import TemperatureScaling
from cognix.communication.top_k import TopKCommunication
from cognix.communication.information_gain import InformationGainRouter
from cognix.explainability.epistemic_shapley import EpistemicShapley
from cognix.decision.escalation import EscalationEngine
from cognix.belief.base import BeliefState

# 1. Uncertainty: MC Dropout
def test_mc_dropout():
    """Test that MC Dropout correctly computes mean and variance over T passes."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # A mock layer that adds noise during training/dropout mode
            if self.training:
                return x + torch.randn_like(x) * 2.0
            return x

    model = DummyModel()
    mc = MonteCarloDropout(num_passes=100, task_type="regression")
    
    # Input is all zeros
    x = torch.zeros(1, 10)
    
    # Run MC Dropout
    est = mc.estimate(model, x)
    mean_out = np.mean(est.raw_samples, axis=0)
    var_out = np.var(est.raw_samples, axis=0)
    
    # Theoretical mean of Gaussian noise is ~0. Variance should be ~4.0 (2.0^2)
    assert np.allclose(mean_out, np.zeros_like(mean_out), atol=0.5), "MC Mean failed"
    assert np.allclose(var_out, np.ones_like(var_out) * 4.0, atol=1.0), "MC Variance failed"

# 2. Uncertainty: Deep Ensembles
def test_deep_ensemble():
    """Test ensemble variance calculation."""
    class DummyModel(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
        def forward(self, x):
            return x + self.offset
            
    # Ensemble members predicting 0, 2, 4
    m1, m2, m3 = DummyModel(0.0), DummyModel(2.0), DummyModel(4.0)
    ensemble = DeepEnsemble(task_type="regression")
    
    x = torch.zeros(1, 1)
    est = ensemble.estimate([m1, m2, m3], x)
    mean_out = np.mean(est.raw_samples, axis=0)
    var_out = np.var(est.raw_samples, axis=0)
    
    # Mean of [0, 2, 4] is 2.0
    # Variance of [0, 2, 4] is ((0-2)^2 + (2-2)^2 + (4-2)^2)/3 = (4 + 0 + 4)/3 = 8/3 ≈ 2.666
    assert np.allclose(mean_out, np.array([[2.0]])), "Ensemble mean failed"
    assert np.allclose(var_out, np.array([[8.0/3.0]]), atol=1e-4), "Ensemble variance failed"

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
    fuser = CognixBeliefFuser()
    
    # Two agents predicting conflicting classes [1, 0] vs [0, 1]
    # Agent 1 has HIGH epistemic uncertainty (0.9), Agent 2 has LOW (0.1)
    b1 = BeliefState("a1", np.array([1.0, 0.0]), alpha=1.0, beta_param=1.0, confidence=1.0)
    b2 = BeliefState("a2", np.array([0.0, 1.0]), alpha=1.0, beta_param=1.0, confidence=1.0)
    
    # Note: Epistemic uncertainties are passed via kwargs in the strategy
    result = fuser.fuse(
        beliefs=[b1, b2], 
        strategy=FusionStrategy.EPISTEMIC_WEIGHTED, 
        epistemic_uncertainties={"a1": 0.9, "a2": 0.1}
    )
    
    # Weight 1 = 1 / 0.9 = 1.11, Weight 2 = 1 / 0.1 = 10.0
    # Expected fused = (1.11 * [1, 0] + 10.0 * [0, 1]) / 11.11 = [0.1, 0.9]
    assert np.allclose(result.belief, np.array([0.1, 0.9]), atol=0.05), "Epistemic fusion failed math check"

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
    cp.calibrate(cal_outputs, cal_labels)
    
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
    
    attribution = es.compute(["a1", "a2"], mock_pipeline, num_samples=100)
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
