"""
COGNIX: A domain-agnostic, uncertainty-aware multi-agent AI decision framework.

Version: 0.1.0
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cognix")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "COGNIX Contributors"
__description__ = "A domain-agnostic, uncertainty-aware multi-agent AI decision framework"

# Core public API
from cognix.engine.decision_engine import DecisionEngine
from cognix.engine.pipeline import CognixPipeline
from cognix.engine.result import DecisionResult, RiskLevel, DecisionOutcome
from cognix.agents.base import StandardAgent
from cognix.agents.registry import AgentRegistry
from cognix.uncertainty.mc_dropout import MCDropout as MonteCarloDropout
from cognix.uncertainty.deep_ensemble import DeepEnsemble
from cognix.uncertainty.decomposition import UncertaintyDecomposition
from cognix.belief.bayesian import BayesianBelief
from cognix.belief.fusion import EpistemicWeightedFusion, AverageFusion
from cognix.calibration.conformal import ConformalPredictor
from cognix.calibration.temperature import TemperatureScaling
from cognix.decision.escalation import EscalationEngine
from cognix.config.schema import CognixConfig
from cognix.graph.standard_gat import StandardGAT
from cognix.graph.epistemic_gat import EpistemicGAT
from cognix.graph.no_graph import NoGraph

__all__ = [
    "__version__",
    "CognixPipeline",
    "DecisionEngine",
    "DecisionResult",
    "RiskLevel",
    "DecisionOutcome",
    "StandardAgent",
    "AgentRegistry",
    "MonteCarloDropout",
    "DeepEnsemble",
    "UncertaintyDecomposition",
    "BayesianBelief",
    "EpistemicWeightedFusion",
    "AverageFusion",
    "ConformalPredictor",
    "TemperatureScaling",
    "EscalationEngine",
    "CognixConfig",
    "StandardGAT",
    "EpistemicGAT",
    "NoGraph",
]

