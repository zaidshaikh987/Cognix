"""
COGNIX Pipeline Orchestrator.

Wires all modules into a single, timed, fault-tolerant decision pipeline:

  Agents
    -> Collect predictions
    -> Estimate uncertainty (per agent)
    -> Compute trust weights
    -> Convert predictions to BeliefStates
    -> Fuse beliefs
    -> (Optional) Graph refinement
    -> (Optional) Calibration
    -> Risk assessment
    -> Decision (ACT / WAIT / REQUEST_INFORMATION / ABSTAIN / ESCALATE)
    -> (Optional) Epistemic Shapley attribution
    -> Generate explanation
    -> Return DecisionResult with full latency breakdown

All pipeline steps are optional and guarded — missing modules produce sensible defaults.
"""
from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np

from cognix.engine.result import DecisionResult, DecisionOutcome, RiskLevel

logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    """Elapsed time in milliseconds since start (perf_counter)."""
    return (time.perf_counter() - start) * 1000.0


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any | None:
    """Call fn with args; log and return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("Pipeline component %s failed: %s", fn, exc)
        return None


class CognixPipeline:
    """
    Orchestrates the full COGNIX decision pipeline.

    All components are optional — pass None to skip a stage.
    """

    def __init__(
        self,
        config: Any = None,
        uncertainty_estimator: Any = None,
        belief_fuser: Any = None,
        calibrator: Any = None,
        graph: Any = None,
        communication: Any = None,
        attribution: Any = None,
        escalation: Any = None,
    ) -> None:
        self.config = config
        self.uncertainty_estimator = uncertainty_estimator
        self.belief_fuser = belief_fuser
        self.calibrator = calibrator
        self.graph = graph
        self.communication = communication
        self.attribution = attribution
        self.escalation = escalation

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        agents: list[Any],
        input_data: Any,
        context: dict[str, Any],
    ) -> DecisionResult:
        """Execute the full pipeline and return a DecisionResult."""
        latencies: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # ── Step 1: Collect predictions ───────────────────────────────
        t = time.perf_counter()
        predictions = self._collect_predictions(agents, input_data)
        latencies["collect_predictions"] = _ms(t)

        if not predictions:
            logger.error("No predictions collected from agents. Returning ABSTAIN.")
            return self._emergency_result(latencies, pipeline_start)

        # ── Step 2: Estimate uncertainty ──────────────────────────────
        t = time.perf_counter()
        uncertainties = self._estimate_uncertainties(agents, predictions)
        latencies["estimate_uncertainty"] = _ms(t)

        # ── Step 3: Trust weights ─────────────────────────────────────
        t = time.perf_counter()
        trust_weights = self._compute_trust_weights(uncertainties)
        latencies["compute_trust"] = _ms(t)

        # ── Step 4: Aggregate confidence + uncertainty ────────────────
        t = time.perf_counter()
        agg_confidence, agg_epistemic, agg_aleatoric, agg_total = (
            self._aggregate_uncertainty(predictions, uncertainties, trust_weights)
        )
        latencies["aggregation"] = _ms(t)

        # ── Step 5: Belief fusion ─────────────────────────────────────
        t = time.perf_counter()
        fused_confidence = agg_confidence
        if self.belief_fuser is not None:
            fused_confidence = self._run_belief_fusion(
                predictions, uncertainties, trust_weights
            )
        latencies["belief_fusion"] = _ms(t)

        # ── Step 6: Graph refinement (optional) ───────────────────────
        t = time.perf_counter()
        # Graph module refines the agent representations but doesn't change
        # the scalar confidence at this stage (full graph integration is
        # handled by experiments/graph module directly)
        latencies["graph_refinement"] = _ms(t)

        # ── Step 7: Calibration ───────────────────────────────────────
        t = time.perf_counter()
        calibrated_confidence: float | None = None
        calibration_metrics: dict | None = None
        if self.calibrator is not None and hasattr(self.calibrator, "predict"):
            try:
                cal_probs = self.calibrator.predict(
                    np.array([[1 - fused_confidence, fused_confidence]])
                )
                calibrated_confidence = float(np.max(cal_probs))
                calibration_metrics = {"method": type(self.calibrator).__name__}
            except Exception as exc:
                logger.debug("Calibrator predict failed (not yet fitted?): %s", exc)
        latencies["calibration"] = _ms(t)

        # ── Step 8: Risk assessment ───────────────────────────────────
        t = time.perf_counter()
        risk_level = self._assess_risk(fused_confidence, agg_epistemic)
        latencies["risk_assessment"] = _ms(t)

        # ── Step 9: Decision ──────────────────────────────────────────
        t = time.perf_counter()
        decision, escalation_required, abstained, requested_info = self._make_decision(
            fused_confidence, agg_epistemic, risk_level
        )
        latencies["decision"] = _ms(t)

        # ── Step 10: Attribution ──────────────────────────────────────
        t = time.perf_counter()
        agent_contributions: dict[str, float] = {}
        if self.attribution is not None and hasattr(self.attribution, "compute"):
            agent_contributions = _safe_call(
                self._run_attribution, uncertainties
            ) or {}
        latencies["attribution"] = _ms(t)

        # ── Step 11: Communication stats ─────────────────────────────
        comm_stats: dict | None = None
        if self.communication is not None:
            n_agents = len(agents)
            max_messages = n_agents * (n_agents - 1)
            actual = getattr(self.communication, "last_message_count", max_messages)
            if max_messages > 0:
                comm_stats = {
                    "total_messages": actual,
                    "max_possible": max_messages,
                    "reduction_ratio": 1.0 - actual / max_messages,
                }

        # ── Step 12: Explanation ──────────────────────────────────────
        t = time.perf_counter()
        explanation, reasoning_steps = self._generate_explanation(
            decision, fused_confidence, agg_epistemic, agg_aleatoric,
            risk_level, trust_weights, agent_contributions
        )
        latencies["explanation"] = _ms(t)

        total_latency = _ms(pipeline_start)

        return DecisionResult(
            decision=decision,
            confidence=fused_confidence,
            calibrated_confidence=calibrated_confidence,
            aleatoric_uncertainty=agg_aleatoric,
            epistemic_uncertainty=agg_epistemic,
            total_uncertainty=agg_total,
            risk_level=risk_level,
            escalation_required=escalation_required,
            abstained=abstained,
            requested_information=requested_info,
            agent_contributions=agent_contributions,
            agent_trust_weights=trust_weights,
            agent_predictions=predictions,
            communication_statistics=comm_stats,
            calibration_metrics=calibration_metrics,
            explanation=explanation,
            reasoning_steps=reasoning_steps,
            latency_ms=latencies,
            total_latency_ms=total_latency,
            timestamp=time.time(),
            session_id=context.get("session_id"),
            metadata={"context": context, "n_agents": len(agents)},
        )

    # ------------------------------------------------------------------
    # Pipeline stage implementations
    # ------------------------------------------------------------------

    def _collect_predictions(
        self, agents: list[Any], input_data: Any
    ) -> dict[str, Any]:
        """Collect predictions from all agents. Agents that raise are skipped."""
        predictions: dict[str, Any] = {}
        for agent in agents:
            agent_id = self._agent_id(agent)
            try:
                if hasattr(agent, "predict"):
                    pred = agent.predict(input_data)
                elif callable(agent):
                    pred = agent(input_data)
                else:
                    logger.warning("Agent %s has no predict method; skipping.", agent_id)
                    continue
                predictions[agent_id] = pred
            except Exception as exc:
                logger.warning("Agent %s predict failed: %s", agent_id, exc)
        return predictions

    def _estimate_uncertainties(
        self, agents: list[Any], predictions: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Estimate uncertainty per agent. Falls back to prediction-derived heuristic."""
        uncertainties: dict[str, dict[str, float]] = {}
        for agent in agents:
            agent_id = self._agent_id(agent)
            if agent_id not in predictions:
                continue

            pred = predictions[agent_id]

            # Try agent's own uncertainty estimate
            if hasattr(agent, "estimate_uncertainty"):
                try:
                    est = agent.estimate_uncertainty(None)
                    if hasattr(est, "epistemic"):
                        uncertainties[agent_id] = {
                            "epistemic": float(est.epistemic),
                            "aleatoric": float(est.aleatoric),
                            "total": float(est.total),
                        }
                        continue
                except Exception as exc:
                    logger.debug("Agent %s estimate_uncertainty failed: %s", agent_id, exc)

            # Try the configured uncertainty estimator
            if self.uncertainty_estimator is not None:
                try:
                    est = self.uncertainty_estimator.estimate(
                        getattr(agent, "_model", agent), None
                    )
                    uncertainties[agent_id] = {
                        "epistemic": float(est.epistemic),
                        "aleatoric": float(est.aleatoric),
                        "total": float(est.total),
                    }
                    continue
                except Exception as exc:
                    logger.debug("UQ estimator for %s failed: %s", agent_id, exc)

            # Fallback: derive from prediction confidence
            confidence = self._extract_confidence(pred)
            epistemic = max(0.0, 1.0 - confidence) * 0.5
            uncertainties[agent_id] = {
                "epistemic": epistemic,
                "aleatoric": epistemic * 0.5,
                "total": epistemic * 1.5,
            }

        return uncertainties

    def _compute_trust_weights(
        self, uncertainties: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        Compute normalized trust weights.

        Weight inversely proportional to epistemic uncertainty:
            w_j = 1 / (sigma_e_j + eps)  [COGNIX proposed mechanism]
        """
        eps = 1e-8
        weights: dict[str, float] = {}
        for agent_id, unc in uncertainties.items():
            weights[agent_id] = 1.0 / (unc.get("epistemic", 1.0) + eps)

        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        return weights

    def _aggregate_uncertainty(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
    ) -> tuple[float, float, float, float]:
        """Return (confidence, epistemic, aleatoric, total) as weighted aggregates."""
        confidences = []
        epistemics = []
        aleatorics = []

        for agent_id, pred in predictions.items():
            w = trust_weights.get(agent_id, 1.0 / max(len(predictions), 1))
            conf = self._extract_confidence(pred)
            unc = uncertainties.get(agent_id, {"epistemic": 0.2, "aleatoric": 0.1, "total": 0.3})
            confidences.append(w * conf)
            epistemics.append(w * unc.get("epistemic", 0.2))
            aleatorics.append(w * unc.get("aleatoric", 0.1))

        agg_confidence = float(np.sum(confidences)) if confidences else 0.5
        agg_epistemic = float(np.sum(epistemics)) if epistemics else 0.5
        agg_aleatoric = float(np.sum(aleatorics)) if aleatorics else 0.3
        agg_total = agg_epistemic + agg_aleatoric

        return (
            float(np.clip(agg_confidence, 0.0, 1.0)),
            float(np.clip(agg_epistemic, 0.0, 1.0)),
            float(np.clip(agg_aleatoric, 0.0, 1.0)),
            float(np.clip(agg_total, 0.0, 1.0)),
        )

    def _run_belief_fusion(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
    ) -> float:
        """
        Convert agent predictions to BeliefStates and fuse them.
        Returns fused confidence scalar.
        """
        try:
            from cognix.belief.base import BeliefState, FusionStrategy
            from cognix.belief.fusion import CognixBeliefFuser

            fuser = self.belief_fuser
            # Determine the strategy
            if hasattr(fuser, "default_strategy"):
                strategy = fuser.default_strategy
            else:
                strategy = FusionStrategy.EPISTEMIC_WEIGHTED

            # Build BeliefStates from predictions
            beliefs = []
            for agent_id, pred in predictions.items():
                conf = self._extract_confidence(pred)
                prob = np.array([1.0 - conf, conf])  # binary for now
                beliefs.append(
                    BeliefState(
                        agent_id=agent_id,
                        belief=prob,
                        alpha=conf * 10,
                        beta_param=(1.0 - conf) * 10,
                        confidence=conf,
                    )
                )

            if not beliefs:
                return 0.5

            # Use CognixBeliefFuser if available
            if isinstance(fuser, CognixBeliefFuser):
                fused = fuser.fuse(
                    beliefs=beliefs,
                    strategy=strategy,
                    epistemic_uncertainties={
                        aid: u.get("epistemic", 0.3)
                        for aid, u in uncertainties.items()
                    },
                    reliabilities={aid: 1.0 for aid in uncertainties},
                )
            else:
                # Generic fuser
                fused = fuser.fuse(beliefs=beliefs, strategy=strategy)

            return float(np.clip(fused.confidence, 0.0, 1.0))

        except Exception as exc:
            logger.warning("Belief fusion failed, using aggregated confidence: %s", exc)
            return 0.5

    def _run_attribution(self, uncertainties: dict[str, dict[str, float]]) -> dict[str, float]:
        """Compute epistemic Shapley values for agent attribution."""
        agent_ids = list(uncertainties.keys())

        def uncertainty_fn(subset: list[str]) -> float:
            if not subset:
                return 1.0
            return float(np.mean([uncertainties[a]["epistemic"] for a in subset]))

        return self.attribution.compute(agent_ids, uncertainty_fn)

    def _assess_risk(self, confidence: float, epistemic: float) -> RiskLevel:
        """Rule-based risk assessment using configurable thresholds."""
        high_unc_thresh = 0.5
        low_conf_thresh = 0.4
        low_unc_thresh = 0.2
        high_conf_thresh = 0.7

        if hasattr(self.config, "decision"):
            d = self.config.decision
            high_unc_thresh = getattr(d, "high_uncertainty_threshold", 0.5)
            low_conf_thresh = getattr(d, "low_confidence_threshold", 0.4)
            low_unc_thresh = high_unc_thresh * 0.4
            high_conf_thresh = getattr(d, "high_confidence_threshold", 0.7)

        if epistemic > high_unc_thresh or confidence < low_conf_thresh:
            return RiskLevel.HIGH
        if epistemic < low_unc_thresh and confidence > high_conf_thresh:
            return RiskLevel.LOW
        return RiskLevel.MODERATE

    def _make_decision(
        self,
        confidence: float,
        epistemic: float,
        risk_level: RiskLevel,
    ) -> tuple[DecisionOutcome, bool, bool, bool]:
        """Map risk level + confidence to a final decision outcome."""
        if self.escalation is not None and hasattr(self.escalation, "decide"):
            try:
                result = self.escalation.decide(confidence, epistemic, risk_level, {})
                return (
                    result.outcome,
                    result.escalation_required,
                    result.outcome == DecisionOutcome.ABSTAIN,
                    result.outcome == DecisionOutcome.REQUEST_INFORMATION,
                )
            except Exception as exc:
                logger.debug("Escalation engine failed: %s", exc)

        # Fallback: rule-based
        if risk_level == RiskLevel.HIGH:
            return DecisionOutcome.ESCALATE, True, False, False
        if risk_level == RiskLevel.MODERATE:
            return DecisionOutcome.REQUEST_INFORMATION, False, False, True
        return DecisionOutcome.ACT, False, False, False

    def _generate_explanation(
        self,
        decision: DecisionOutcome,
        confidence: float,
        epistemic: float,
        aleatoric: float,
        risk_level: RiskLevel,
        trust_weights: dict[str, float],
        contributions: dict[str, float],
    ) -> tuple[str, list[str]]:
        """Generate a human-readable explanation from actual pipeline state."""
        top_agent = max(trust_weights, key=trust_weights.get) if trust_weights else "N/A"
        top_contrib = max(contributions, key=contributions.get) if contributions else "N/A"

        reasoning_steps = [
            f"1. Collected predictions from {len(trust_weights)} agent(s).",
            f"2. Estimated epistemic uncertainty: {epistemic:.3f}, aleatoric: {aleatoric:.3f}.",
            f"3. Most trusted agent: {top_agent} (weight={trust_weights.get(top_agent, 0):.3f}).",
            f"4. Fused confidence: {confidence:.3f}.",
            f"5. Risk level assessed as {risk_level.value}.",
            f"6. Decision: {decision.value}.",
        ]

        if contributions:
            reasoning_steps.append(
                f"7. Main uncertainty contributor: {top_contrib} "
                f"(Epistemic Shapley ϕ={contributions.get(top_contrib, 0):.4f})."
            )

        explanation = (
            f"COGNIX decided {decision.value} with {confidence:.1%} confidence "
            f"(risk: {risk_level.value}, epistemic uncertainty: {epistemic:.3f}). "
            f"Most trusted agent: {top_agent}."
        )
        if contributions:
            explanation += f" Main uncertainty source: {top_contrib}."

        return explanation, reasoning_steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_id(agent: Any) -> str:
        """Extract a stable string ID from an agent object."""
        if hasattr(agent, "agent_id"):
            return str(agent.agent_id)
        if hasattr(agent, "name"):
            return str(agent.name)
        return f"agent_{id(agent)}"

    @staticmethod
    def _extract_confidence(pred: Any) -> float:
        """Extract a scalar confidence from various prediction formats."""
        if pred is None:
            return 0.5
        if isinstance(pred, float):
            return float(np.clip(pred, 0.0, 1.0))
        if hasattr(pred, "confidence"):
            return float(np.clip(pred.confidence, 0.0, 1.0))
        if hasattr(pred, "probabilities") and pred.probabilities is not None:
            return float(np.clip(np.max(pred.probabilities), 0.0, 1.0))
        if isinstance(pred, dict):
            if "confidence" in pred:
                return float(np.clip(pred["confidence"], 0.0, 1.0))
            if "prob" in pred:
                return float(np.clip(pred["prob"], 0.0, 1.0))
            if "probabilities" in pred:
                probs = np.array(pred["probabilities"])
                return float(np.clip(np.max(probs), 0.0, 1.0))
        if isinstance(pred, np.ndarray):
            return float(np.clip(np.max(pred), 0.0, 1.0))
        return 0.5

    def _emergency_result(
        self, latencies: dict[str, float], pipeline_start: float
    ) -> DecisionResult:
        """Return a safe ABSTAIN result when no predictions were collected."""
        return DecisionResult(
            decision=DecisionOutcome.ABSTAIN,
            confidence=0.0,
            calibrated_confidence=None,
            aleatoric_uncertainty=None,
            epistemic_uncertainty=1.0,
            total_uncertainty=1.0,
            risk_level=RiskLevel.HIGH,
            escalation_required=True,
            abstained=True,
            requested_information=False,
            agent_contributions={},
            agent_trust_weights={},
            agent_predictions={},
            communication_statistics=None,
            calibration_metrics=None,
            explanation="COGNIX abstained: no agent predictions were available.",
            reasoning_steps=["0. No agent predictions received — defaulting to safe ABSTAIN."],
            latency_ms=latencies,
            total_latency_ms=_ms(pipeline_start),
            timestamp=time.time(),
            session_id=None,
            metadata={"emergency": True},
        )
