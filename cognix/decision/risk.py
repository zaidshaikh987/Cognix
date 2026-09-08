"""
Risk Assessment.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

@dataclass
class RiskAssessment:
    risk_level: RiskLevel
    risk_score: float
    reasoning: str
    component_risks: dict[str, float]

class RiskAssessor:
    def __init__(self, high_epistemic_threshold: float = 0.5, low_confidence_threshold: float = 0.4, low_epistemic_threshold: float = 0.2, high_confidence_threshold: float = 0.7):
        # Default thresholds (documented as defaults, not ground truth)
        self.high_epistemic_threshold = high_epistemic_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.low_epistemic_threshold = low_epistemic_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def assess(self, confidence: float, epistemic_uncertainty: float, aleatoric_uncertainty: float, ood_score: Optional[float] = None) -> RiskAssessment:
        risk_score = (epistemic_uncertainty + (1.0 - confidence)) / 2.0
        component_risks = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "confidence_gap": 1.0 - confidence
        }
        if ood_score is not None:
            component_risks["ood"] = ood_score
            risk_score = (risk_score + ood_score) / 2.0

        if epistemic_uncertainty > self.high_epistemic_threshold or confidence < self.low_confidence_threshold:
            level = RiskLevel.HIGH
            reason = "High epistemic uncertainty or low confidence."
        elif epistemic_uncertainty < self.low_epistemic_threshold and confidence > self.high_confidence_threshold:
            level = RiskLevel.LOW
            reason = "Low epistemic uncertainty and high confidence."
        else:
            level = RiskLevel.MODERATE
            reason = "Moderate risk levels."
            
        return RiskAssessment(
            risk_level=level,
            risk_score=min(max(risk_score, 0.0), 1.0),
            reasoning=reason,
            component_risks=component_risks
        )
