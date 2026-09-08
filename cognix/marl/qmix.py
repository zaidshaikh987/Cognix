"""
QMIX: Monotonic Value Function Factorisation.
Reference: Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

class QMIXAgent(nn.Module):
    """Decentralized DRQN Agent."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden)
        q_values = self.fc2(h)
        return q_values, h

class QMIXMixer(nn.Module):
    """
    Centralized Mixer network with hypernetworks.
    Ensures monotonicity: dQ_tot / dQ_a >= 0 by taking absolute value of weights.
    """
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        # Hypernetworks generate weights from global state
        self.hyper_w1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        
        # Biases
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (batch_size, n_agents)
        states: (batch_size, state_dim)
        """
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # Monotonic weights
        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.embed_dim, 1)
        
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        
        # Mixer operations
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(bs, -1)

class QMIX:
    """Wrapper for the full QMIX algorithm."""
    def __init__(self, agents: list[QMIXAgent], mixer: QMIXMixer):
        self.agents = agents
        self.mixer = mixer
