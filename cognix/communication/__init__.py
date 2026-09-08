"""
COGNIX Communication Routing Strategies.
"""
from .top_k import TopKCommunication
from .information_gain import InformationGainRouter

__all__ = ["TopKCommunication", "InformationGainRouter"]
