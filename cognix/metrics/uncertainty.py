import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

def uncertainty_error_correlation(uncertainties: np.ndarray, errors: np.ndarray) -> float:
    corr, _ = spearmanr(uncertainties, errors)
    return float(corr)

def auroc_ood(in_dist_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([in_dist_scores, ood_scores])
    return float(roc_auc_score(y_true, y_scores))

def auprc_ood(in_dist_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([in_dist_scores, ood_scores])
    return float(average_precision_score(y_true, y_scores))

def selective_risk(predictions: np.ndarray, labels: np.ndarray, confidences: np.ndarray, coverage_levels: list) -> dict:
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_preds = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    n = len(predictions)
    risks = {}
    for cov in coverage_levels:
        cutoff = int(n * cov)
        if cutoff == 0:
            risks[cov] = 0.0
            continue
        cov_preds = sorted_preds[:cutoff]
        cov_labels = sorted_labels[:cutoff]
        risk = float(np.mean(cov_preds != cov_labels))
        risks[cov] = risk
        
    return risks

def uncertainty_decomposition_consistency(aleatoric: np.ndarray, epistemic: np.ndarray, total: np.ndarray) -> bool:
    # Check if epistemic + aleatoric is roughly equal to total
    diff = np.abs(total - (aleatoric + epistemic))
    return bool(np.all(diff < 1e-4))
