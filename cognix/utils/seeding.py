"""Reproducibility utilities — random seed management."""
import random
import numpy as np
from typing import Optional


def set_random_seed(seed: int, torch_deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across numpy, random, and optionally torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy Generator with optional seed."""
    return np.random.default_rng(seed)
