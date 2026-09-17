# Risk-Aware Decisions

Multi-agent reasoning and belief fusion ultimately culminate in a single action. But what happens if the system is still uncertain?

COGNIX provides an `EscalationEngine` to intercept unsafe decisions before they occur.

## EscalationEngine

**Import**: `from cognix import EscalationEngine`

The Escalation Engine takes the final outputs of the belief fusion and calibration steps and checks them against strict safety policies. It evaluates:
1. **Confidence Threshold**: Is the calibrated confidence below a safe minimum?
2. **Epistemic Threshold**: Is the total epistemic uncertainty unacceptably high (indicating severe OOD conditions)?
3. **Conformal Set Size**: Did the Conformal Predictor return a set of 3 or more potential outcomes? (This indicates total confusion).

If any of these conditions are met, the engine overrides the standard `ACT` outcome and returns an `ESCALATE` signal.

```python
from cognix import EscalationEngine
from cognix import RiskLevel, DecisionOutcome

engine = EscalationEngine()

result = engine.evaluate(
    confidence=0.82,
    epistemic_uncertainty=0.06,
    conformal_set_size=1
)

if result.escalation == True:
    print("Handing control to human operator!")
else:
    print("Safe to act autonomously.")
```

## Explainability vs. Escalation

COGNIX computes **Epistemic Shapley** values to attribute the total epistemic uncertainty back to the individual agents. 

**IMPORTANT**: Shapley attribution is implemented for **explanation only**. An agent having a high Shapley value does *not* trigger the `EscalationEngine`. Only the macroscopic system states (Confidence, Total Epistemic Uncertainty, and Conformal Set Size) are used for routing decisions.
