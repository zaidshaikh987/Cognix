"""
Selective Prediction / Abstention.
Reference concept: Chow, C. K. (1957). An optimum character recognition system using decision functions.
"""
from dataclasses import dataclass
from typing import Any, Optional
from cognix.decision.risk import RiskLevel

@dataclass
class AbstentionDecision:
    abstain: bool
    reason: str
    recommended_action: str  # 'ACT'/'WAIT'/'REQUEST_INFORMATION'/'ABSTAIN'/'ESCALATE'

class AbstentionPolicy:
    def __init__(self, risk_tolerance: RiskLevel = RiskLevel.MODERATE):
        self.risk_tolerance = risk_tolerance

    def should_abstain(self, confidence: float, epistemic_uncertainty: float, risk_level: RiskLevel, context: Optional[dict[str, Any]] = None) -> AbstentionDecision:
        if risk_level == RiskLevel.HIGH:
            return AbstentionDecision(
                abstain=True,
                reason="Risk level is HIGH, abstaining from decision.",
                recommended_action="ESCALATE"
            )
        elif risk_level == RiskLevel.MODERATE and self.risk_tolerance == RiskLevel.LOW:
            return AbstentionDecision(
                abstain=True,
                reason="Risk level is MODERATE but tolerance is LOW.",
                recommended_action="REQUEST_INFORMATION"
            )
            
        return AbstentionDecision(
            abstain=False,
            reason="Risk is within acceptable bounds.",
            recommended_action="ACT"
        )
