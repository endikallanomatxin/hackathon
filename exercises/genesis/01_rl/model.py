import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 from_checkpoint=None):
        super().__init__()

        internal_dim = 256

        # Common layers (2)
        prev = obs_dim
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(prev, internal_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.LayerNorm(internal_dim))
            prev = internal_dim
        self.common = nn.Sequential(*layers)

        # Action layers (4)
        action_layers = []
        for _ in range(4):
            action_layers.append(nn.Linear(internal_dim, internal_dim))
            action_layers.append(nn.LeakyReLU())
            action_layers.append(nn.LayerNorm(internal_dim))
        self.action = nn.Sequential(*action_layers)
        self.action_mean = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(internal_dim),
            nn.Linear(internal_dim, act_dim),
            nn.Tanh()
        )
        self.action_std = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(internal_dim),
            nn.Linear(internal_dim, act_dim),
            nn.Softplus()
        )

        # Value layers (4)
        value_layers = []
        for _ in range(4):
            value_layers.append(nn.Linear(internal_dim, internal_dim))
            value_layers.append(nn.LeakyReLU())
            value_layers.append(nn.LayerNorm(internal_dim))
        self.value = nn.Sequential(*value_layers, nn.Linear(internal_dim, 1))

        if from_checkpoint is not None:
            self.load_state_dict(torch.load(from_checkpoint))

    def forward(self, obs):
        # obs: tensor de forma [B, obs_dim]
        # Returns the logits of the actions mean and std
        # shape [B, act_dim]
        common = self.common(obs)
        action = self.action(common)
        action_mean = self.action_mean(action)
        action_std = self.action_std(action)
        value = self.value(common)
        value = value.squeeze(-1)
        return action_mean, action_std, value
