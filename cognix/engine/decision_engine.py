"""
COGNIX Decision Engine — public API entry point.

Usage::

    from cognix import DecisionEngine

    engine = DecisionEngine(
        uncertainty='mc_dropout',
        belief='epistemic_weighted',
        calibrator='conformal',
        config=CognixConfig(),
    )
    result = engine.decide(agents=[...], input_data=data, context={})
    print(result.summary())
"""
from __future__ import annotations

import logging
from typing import Any

from cognix.engine.pipeline import CognixPipeline
from cognix.engine.result import DecisionResult

logger = logging.getLogger(__name__)


def _build_uncertainty(spec: Any) -> Any:
    """Resolve uncertainty estimator from string shorthand or return object as-is."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.lower()
    if spec == "mc_dropout":
        from cognix.uncertainty.mc_dropout import MonteCarloDropout
        return MonteCarloDropout()
    if spec == "deep_ensemble":
        from cognix.uncertainty.deep_ensemble import DeepEnsemble
        return DeepEnsemble()
    raise ValueError(
        f"Unknown uncertainty method: {spec!r}. "
        "Valid strings: 'mc_dropout', 'deep_ensemble'. "
        "Or pass an UncertaintyEstimator instance directly."
    )


def _build_belief(spec: Any) -> Any:
    """Resolve belief fuser from string shorthand or return object as-is."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    from cognix.belief.base import FusionStrategy
    from cognix.belief.fusion import CognixBeliefFuser
    mapping = {
        "uniform": FusionStrategy.UNIFORM,
        "majority": FusionStrategy.MAJORITY,
        "confidence": FusionStrategy.CONFIDENCE,
        "reliability": FusionStrategy.RELIABILITY,
        "epistemic_weighted": FusionStrategy.EPISTEMIC_WEIGHTED,
        "bayesian": FusionStrategy.BAYESIAN,
    }
    strategy_key = spec.lower()
    if strategy_key not in mapping:
        raise ValueError(
            f"Unknown belief fusion method: {spec!r}. "
            f"Valid strings: {list(mapping.keys())}"
        )
    fuser = CognixBeliefFuser()
    fuser.default_strategy = mapping[strategy_key]
    return fuser


def _build_calibrator(spec: Any) -> Any:
    """Resolve calibrator from string shorthand or return object as-is."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.lower()
    if spec == "conformal":
        from cognix.calibration.conformal import ConformalPredictor
        return ConformalPredictor()
    if spec == "temperature":
        from cognix.calibration.temperature import TemperatureScaling
        return TemperatureScaling()
    raise ValueError(f"Unknown calibrator: {spec!r}. Valid strings: 'conformal', 'temperature'.")


def _build_communication(spec: Any) -> Any:
    """Resolve communication protocol from string shorthand."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.lower()
    if spec == "top_k":
        from cognix.communication.top_k import TopKCommunication
        return TopKCommunication()
    if spec == "information_gain":
        from cognix.communication.information_gain import InformationGainRouter
        return InformationGainRouter()
    raise ValueError(f"Unknown communication method: {spec!r}. Valid: 'top_k', 'information_gain'.")


def _build_attribution(spec: Any) -> Any:
    """Resolve attribution from string shorthand."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.lower()
    if spec == "epistemic_shapley":
        from cognix.explainability.epistemic_shapley import EpistemicShapley
        return EpistemicShapley()
    if spec == "shap":
        from cognix.explainability.shap_adapter import SHAPAdapter
        return SHAPAdapter()
    raise ValueError(f"Unknown attribution method: {spec!r}. Valid: 'epistemic_shapley', 'shap'.")


def _build_graph(spec: Any) -> Any:
    """Resolve graph module from string shorthand."""
    if spec is None:
        return None
    if not isinstance(spec, str):
        return spec
    spec = spec.lower()
    if spec == "epistemic_gat":
        from cognix.graph.epistemic_gat import EpistemicGAT
        return EpistemicGAT()
    if spec == "gat":
        from cognix.graph.gat import GATNetwork
        return GATNetwork()
    if spec == "gcn":
        from cognix.graph.gcn import GCNNetwork
        return GCNNetwork()
    raise ValueError(f"Unknown graph method: {spec!r}. Valid: 'epistemic_gat', 'gat', 'gcn'.")


def _build_escalation(spec: Any) -> Any:
    """Resolve escalation engine."""
    if spec is None:
        return None
    if isinstance(spec, bool) and spec:
        from cognix.decision.escalation import EscalationEngine
        return EscalationEngine()
    if isinstance(spec, bool) and not spec:
        return None
    if not isinstance(spec, str):
        return spec
    from cognix.decision.escalation import EscalationEngine
    return EscalationEngine()


class DecisionEngine:
    """
    COGNIX Decision Engine — the primary public API.

    Accepts string shorthand or object instances for each module.
    Builds and wires the CognixPipeline internally.

    Example::

        engine = DecisionEngine(
            uncertainty='mc_dropout',
            belief='epistemic_weighted',
            calibrator='conformal',
            escalation=True,
        )
        result = engine.decide(agents=my_agents, input_data=data)
        print(result.summary())
    """

    def __init__(
        self,
        config: Any = None,
        uncertainty: Any = None,
        belief: Any = None,
        calibrator: Any = None,
        communication: Any = None,
        attribution: Any = None,
        escalation: Any = None,
        graph: Any = None,
    ) -> None:
        from cognix.config.schema import CognixConfig
        self.config = config or CognixConfig()

        self.uncertainty = _build_uncertainty(uncertainty)
        self.belief = _build_belief(belief)
        self.calibrator = _build_calibrator(calibrator)
        self.communication = _build_communication(communication)
        self.attribution = _build_attribution(attribution)
        self.escalation_module = _build_escalation(escalation)
        self.graph = _build_graph(graph)

        self.pipeline = CognixPipeline(
            config=self.config,
            uncertainty_estimator=self.uncertainty,
            belief_fuser=self.belief,
            calibrator=self.calibrator,
            graph=self.graph,
            communication=self.communication,
            attribution=self.attribution,
            escalation=self.escalation_module,
        )

        logger.info(
            "COGNIX DecisionEngine initialized | "
            "uncertainty=%s | belief=%s | calibrator=%s | graph=%s | escalation=%s",
            type(self.uncertainty).__name__ if self.uncertainty else "None",
            type(self.belief).__name__ if self.belief else "None",
            type(self.calibrator).__name__ if self.calibrator else "None",
            type(self.graph).__name__ if self.graph else "None",
            type(self.escalation_module).__name__ if self.escalation_module else "None",
        )

    def decide(
        self,
        agents: list[Any],
        input_data: Any,
        context: dict[str, Any] | None = None,
    ) -> DecisionResult:
        """Run the full COGNIX pipeline and return a structured DecisionResult."""
        return self.pipeline.run(agents, input_data, context or {})

    def configure(self, config: Any) -> None:
        """Update the engine configuration."""
        self.config = config
        self.pipeline.config = config

    def __enter__(self) -> "DecisionEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"DecisionEngine("
            f"uncertainty={type(self.uncertainty).__name__ if self.uncertainty else None}, "
            f"belief={type(self.belief).__name__ if self.belief else None}, "
            f"calibrator={type(self.calibrator).__name__ if self.calibrator else None})"
        )
