import numpy as np

def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)
    Reference: Guo et al. (2017) On Calibration of Modern Neural Networks.
    Group predictions into M bins, compute weighted average of |conf - acc|
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for b in range(1, len(bins)):
        mask = (indices == b)
        if np.any(mask):
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += bin_weight * np.abs(bin_conf - bin_acc)
            
    return float(ece)

def reliability_diagram_data(confidences: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """
    Compute data for a reliability diagram.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    
    bin_conf = []
    bin_acc = []
    bin_counts = []
    
    for b in range(1, len(bins)):
        mask = (indices == b)
        count = np.sum(mask)
        bin_counts.append(int(count))
        if count > 0:
            bin_conf.append(float(np.mean(confidences[mask])))
            bin_acc.append(float(np.mean(labels[mask])))
        else:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            
    return {
        'bin_conf': bin_conf,
        'bin_acc': bin_acc,
        'bin_counts': bin_counts,
        'bins': bins.tolist()
    }

def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier score: Mean squared difference between predicted probabilities and actual outcomes.
    labels should be one-hot encoded or integer labels if probabilities are multiclass.
    """
    if len(labels.shape) == 1:
        # Convert to one-hot
        num_classes = probabilities.shape[1]
        labels_one_hot = np.eye(num_classes)[labels]
    else:
        labels_one_hot = labels
        
    return float(np.mean(np.sum((probabilities - labels_one_hot)**2, axis=1)))

def overconfidence_error(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 15) -> float:
    """
    Overconfidence Error (OE): Penalizes predictions where confidence is greater than accuracy.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    
    oe = 0.0
    total_samples = len(confidences)
    
    for b in range(1, len(bins)):
        mask = (indices == b)
        if np.any(mask):
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            bin_weight = np.sum(mask) / total_samples
            # Only penalize if confidence > accuracy
            oe += bin_weight * (bin_conf * np.maximum(bin_conf - bin_acc, 0))
            
    return float(oe)
