import pytest
import numpy as np

def update_belief(prior, likelihood):
    post = prior * likelihood
    return post / np.sum(post)

def fuse_uniform(beliefs):
    fused = np.mean(beliefs, axis=0)
    return fused / np.sum(fused)

def fuse_epistemic_weighted(beliefs, uncertainties):
    weights = 1.0 / (np.array(uncertainties) + 1e-5)
    weights /= np.sum(weights)
    fused = np.zeros_like(beliefs[0])
    for w, b in zip(weights, beliefs):
        fused += w * b
    return fused

def propagate_belief(iters=10):
    val = 0.5
    for _ in range(iters):
        val = val * 0.9 + 0.1
    return val

def test_bayesian_belief_update_increases_confidence():
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.8, 0.2])
    post = update_belief(prior, likelihood)
    assert post[0] > 0.5
    assert post[1] < 0.5

def test_belief_fusion_uniform_weights_equal():
    b1 = np.array([0.8, 0.2])
    b2 = np.array([0.2, 0.8])
    fused = fuse_uniform([b1, b2])
    np.testing.assert_allclose(fused, [0.5, 0.5])

def test_belief_fusion_epistemic_weighted_low_uncertainty_gets_high_weight():
    b1 = np.array([0.9, 0.1]) # low unc
    b2 = np.array([0.1, 0.9]) # high unc
    u1 = 0.1
    u2 = 1.0
    fused = fuse_epistemic_weighted([b1, b2], [u1, u2])
    assert fused[0] > 0.5

def test_belief_propagation_convergence():
    val1 = propagate_belief(10)
    val2 = propagate_belief(20)
    assert abs(val1 - val2) < 0.2

def test_beta_distribution_mean_variance():
    alpha, beta = 2, 2
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    assert mean == 0.5
    assert var > 0

def test_fusion_strategies_produce_valid_probabilities():
    b1 = np.array([0.3, 0.7])
    b2 = np.array([0.6, 0.4])
    fused = fuse_uniform([b1, b2])
    assert np.isclose(np.sum(fused), 1.0)
    assert np.all(fused >= 0)
