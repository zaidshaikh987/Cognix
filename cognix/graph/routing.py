"""
Dynamic graph construction and routing.
"""
import numpy as np

def build_fully_connected_graph(agent_ids: list[str]) -> np.ndarray:
    N = len(agent_ids)
    return np.ones((N, N))

def build_proximity_graph(agent_ids: list[str], positions: dict[str, tuple[float, float]], radius: float) -> np.ndarray:
    N = len(agent_ids)
    A = np.zeros((N, N))
    for i, id_i in enumerate(agent_ids):
        for j, id_j in enumerate(agent_ids):
            if i == j:
                A[i, j] = 1.0
                continue
            pos_i = np.array(positions[id_i])
            pos_j = np.array(positions[id_j])
            dist = np.linalg.norm(pos_i - pos_j)
            if dist <= radius:
                A[i, j] = 1.0
    return A

def top_k_routing(scores: dict[str, float], k: int) -> list[str]:
    sorted_agents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_agents[:k]

def bandwidth_constrained_routing(scores: dict[str, float], max_messages: int) -> list[str]:
    return top_k_routing(scores, max_messages)

def information_gain_score(agent_id: str, epistemic_uncertainty: float, confidence: float) -> float:
    """
    Note: This is a heuristic. More rigorous info-gain requires estimating uncertainty reduction.
    """
    eps = 1e-6
    return confidence / (epistemic_uncertainty + eps)
