import numpy as np

def expected_calibration_error(y_conf: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_conf, bins, right=True)
    
    ece = 0.0
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_conf[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_conf)
    return float(ece)

def reliability_diagram(y_conf: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_conf, bins, right=True)
    
    accs = []
    confs = []
    counts = []
    
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if np.any(mask):
            accs.append(float(np.mean(y_true[mask])))
            confs.append(float(np.mean(y_conf[mask])))
            counts.append(int(np.sum(mask)))
        else:
            accs.append(0.0)
            confs.append(0.0)
            counts.append(0)
            
    return {"accuracies": accs, "confidences": confs, "counts": counts, "bins": bins.tolist()}

def maximum_calibration_error(y_conf: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_conf, bins, right=True)
    
    mce = 0.0
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_conf[mask])
            mce = max(mce, np.abs(acc - conf))
    return float(mce)

def coverage_probability(prediction_sets: list, y_true: np.ndarray) -> float:
    covered = sum(1 for p_set, true_val in zip(prediction_sets, y_true) if true_val in p_set)
    return covered / len(y_true) if len(y_true) > 0 else 0.0

def average_set_size(prediction_sets: list) -> float:
    return sum(len(p_set) for p_set in prediction_sets) / len(prediction_sets) if prediction_sets else 0.0

def brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))
