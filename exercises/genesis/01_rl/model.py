import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 from_checkpoint=None,
                 num_tokens=16,
                 token_dim=128,
                 num_layers=4,
                 num_heads=8,
                 mlp_ratio=4):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        token_space = num_tokens * token_dim

        # Proyecta la observación a una secuencia de tokens aprendidos
        self.obs_to_tokens = nn.Sequential(
            nn.Linear(obs_dim, token_space),
            nn.GELU(),
            nn.Linear(token_space, token_space),
        )
        self.token_norm = nn.LayerNorm(token_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * mlp_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Capas posteriores tras aplanar todos los tokens
        self.post_tokens = nn.Sequential(
            nn.LayerNorm(token_space),
            nn.Linear(token_space, token_space),
            nn.GELU(),
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(token_space),
            nn.Linear(token_space, token_dim),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(token_space),
            nn.Linear(token_space, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

        self.action_mean = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, act_dim),
            nn.Tanh(),
        )
        self.action_std = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, act_dim),
            nn.Softplus(),
        )

        if from_checkpoint is not None:
            self.load_state_dict(torch.load(from_checkpoint))

    def forward(self, obs):
        # obs: tensor de forma [B, obs_dim]
        batch_size = obs.shape[0]

        tokens = self.obs_to_tokens(obs)
        tokens = tokens.view(batch_size, self.num_tokens, self.token_dim)
        tokens = self.token_norm(tokens)
        tokens = tokens + self.pos_embedding
        tokens = self.transformer(tokens)

        tokens_flat = tokens.reshape(batch_size, -1)
        tokens_flat = self.post_tokens(tokens_flat)

        policy_latent = self.policy_head(tokens_flat)
        action_mean = self.action_mean(policy_latent)
        action_std = self.action_std(policy_latent)

        value = self.value_head(tokens_flat).squeeze(-1)

        return action_mean, action_std, value
