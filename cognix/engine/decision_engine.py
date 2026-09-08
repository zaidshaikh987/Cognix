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
        mode: str = "production",
    ) -> None:
        from cognix.config.schema import CognixConfig
        self.config = config or CognixConfig()
        self.mode = mode

        self.uncertainty = uncertainty
        self.belief = belief
        self.calibrator = calibrator
        self.communication = communication
        self.attribution = attribution
        self.escalation_module = escalation
        self.graph = graph

        self.pipeline = CognixPipeline(
            config=self.config,
            uncertainty_estimator=self.uncertainty,
            belief_fuser=self.belief,
            calibrator=self.calibrator,
            graph=self.graph,
            communication=self.communication,
            attribution=self.attribution,
            escalation=self.escalation_module,
            mode=mode,
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
        agent_reliabilities: dict[str, float] | None = None,
    ) -> DecisionResult:
        """Run the full COGNIX pipeline and return a structured DecisionResult."""
        return self.pipeline.run(
            agents, input_data, context or {},
            agent_reliabilities=agent_reliabilities,
        )

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
