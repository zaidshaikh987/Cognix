"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient).
Reference: Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # Continuous action in [-1, 1]
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Centralized Critic: takes all observations and all actions."""
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, obs_dim: int, action_dim: int, total_obs_dim: int, total_action_dim: int, lr: float = 1e-3):
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(total_obs_dim, total_action_dim)
        self.target_critic = Critic(total_obs_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update_targets(self, tau: float = 0.01):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class MADDPG:
    """
    Controller for multiple DDPG agents.
    Implements Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, agents: list[DDPGAgent]):
        self.agents = agents
        
    def step(self, obs_list: list[torch.Tensor], explore: bool = True) -> list[torch.Tensor]:
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.actor(obs_list[i])
            if explore:
                action += torch.randn_like(action) * 0.1 # Exploration noise
            actions.append(torch.clamp(action, -1.0, 1.0))
        return actions
