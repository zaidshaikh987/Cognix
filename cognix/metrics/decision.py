import numpy as np

def escalation_precision_recall_f1(escalation_decisions: np.ndarray, true_escalations: np.ndarray) -> dict:
    tp = np.sum((escalation_decisions == 1) & (true_escalations == 1))
    fp = np.sum((escalation_decisions == 1) & (true_escalations == 0))
    fn = np.sum((escalation_decisions == 0) & (true_escalations == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

def abstention_rate(decisions: list) -> float:
    # Assuming decisions contains enum values like "ABSTAIN"
    abstentions = sum(1 for d in decisions if hasattr(d, 'name') and d.name == "ABSTAIN" or d == "ABSTAIN")
    return abstentions / len(decisions) if decisions else 0.0

def unsafe_action_rate(decisions: list, safe_decisions: list) -> float:
    unsafe = sum(1 for d, s in zip(decisions, safe_decisions) if d == "ACT" and not s)
    act_total = sum(1 for d in decisions if d == "ACT")
    return unsafe / act_total if act_total > 0 else 0.0

def risk_sensitive_utility(decisions: list, outcomes: list, risk_weights: dict) -> float:
    utility = 0.0
    for d, o in zip(decisions, outcomes):
        utility += risk_weights.get((d, o), 0.0)
    return float(utility / len(decisions)) if decisions else 0.0

def decision_accuracy(decisions: list, correct_decisions: list) -> float:
    correct = sum(1 for d, c in zip(decisions, correct_decisions) if d == c)
    return float(correct / len(decisions)) if decisions else 0.0
