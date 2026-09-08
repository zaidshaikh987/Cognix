import time
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from abc import ABC, abstractmethod

class FusionStrategy(Enum):
    UNIFORM = auto()
    MAJORITY = auto()
    CONFIDENCE = auto()
    RELIABILITY = auto()
    EPISTEMIC_WEIGHTED = auto()
    BAYESIAN = auto()

@dataclass
class BeliefState:
    agent_id: str
    belief: np.ndarray
    alpha: float
    beta_param: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True

class BeliefFuser(ABC):
    @abstractmethod
    def fuse(self, beliefs: list[BeliefState], strategy: FusionStrategy, **kwargs) -> BeliefState:
        pass
