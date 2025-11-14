# rl/agent.py
import torch
import torch.optim as optim
import genesis as gs

from model import PolicyNetwork

class PPOAgent:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 lr=3e-5,
                 lr_min=2e-6,
                 lr_t0=40,
                 clip_epsilon=0.2,
                 gamma=0.99,
                 update_epochs=8,
                 from_checkpoint=None
    ):
        self.policy = PolicyNetwork(obs_dim, act_dim, from_checkpoint=from_checkpoint).to('cuda')
        param_count = sum(p.numel() for p in self.policy.parameters())
        gs.logger.info(f"PolicyNetwork has {param_count} parameters")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=lr_t0, T_mult=1, eta_min=lr_min)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.update_epochs = update_epochs

    def select_action_and_get_value(self, obs):
        # obs: tensor de forma [B, obs_dim]
        action_mean, action_std, value = self.policy.forward(obs)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob.sum(dim=-1), value

    def compute_returns(self, rewards):
        """
        rewards: tensores de forma [T, B]
        Calcula los retornos descontados por cada entorno.
        """
        T, B = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            next_return = rewards[t] + self.gamma * next_return
            returns[t] = next_return / (T - t)
        return returns

    def update(self, rollout):
        """
        rollout: diccionario con claves:
            'obs':      [T, B, obs_dim]
            'actions':  [T, B, act_dim]
            'log_probs':[T, B]
            'values':   [T, B]
            'rewards':  [T, B]
        """
        # Extraer tensores del rollout
        obs = rollout['obs']            # [T, B, obs_dim]
        actions = rollout['actions']    # [T, B, act_dim]
        old_log_probs = rollout['log_probs'] # [T, B]
        values = rollout['values']      # [T, B]
        rewards = rollout['rewards']    # [T, B]

        # Calcular retornos descontados para cada entorno
        returns = self.compute_returns(rewards)  # [T, B]
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Aplanar tensores de [T, B, ...] a [T*B, ...]
        T, B = rewards.shape
        obs = obs.view(T * B, -1)
        actions = actions.view(T * B, -1)
        old_log_probs = old_log_probs.view(T * B)
        returns = returns.view(T * B)

        # Calcular ventajas
        advantages = returns - values.view(-1)
        advantages = advantages / (advantages.std() + 1e-8)

        # Mover tensores a CUDA (si no lo están)
        obs = obs.to('cuda')
        actions = actions.to('cuda')
        old_log_probs = old_log_probs.to('cuda')
        advantages = advantages.to('cuda')
        returns = returns.to('cuda')

        target_entropy = actions.shape[-1]  # Entropía objetivo para la bonificación
        # Hiperparámetros para balancear las pérdidas
        value_coef = 0.1      # Coeficiente para la pérdida del valor
        entropy_coef = 0.001  # Coeficiente para la bonificación de entropía

        loss_value = 0.0
        for i in range(self.update_epochs):
            self.optimizer.zero_grad()
            # Forward pass para obtener parámetros de la política y el valor
            mean, std, value_pred = self.policy(obs)  # value_pred shape: [T*B, 1]
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Calcular el ratio entre nueva y antigua probabilidad
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()

            # Pérdida de la política (clipped surrogate objective)
            policy_loss_batch = -torch.min(surr1, surr2)  # Shape [T*B]
            policy_loss_contrib = policy_loss_batch.mean()

            # Pérdida del valor (error cuadrático medio)
            value_loss_batch = torch.nn.functional.mse_loss(value_pred.squeeze(-1), returns, reduction='none')
            value_loss_contrib = value_coef * value_loss_batch.mean()

            # Pérdida por entropía
            entropy = dist.entropy().sum(dim=-1)  # Shape [T*B]
            entropy_loss_batch = - (entropy - target_entropy)  # Shape [T*B]
            entropy_loss_contrib = entropy_coef * entropy_loss_batch.mean()

            # Pérdida total (con bonificación de entropía)
            loss = policy_loss_contrib + value_loss_contrib + entropy_loss_contrib
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_value = loss.item()
            gs.logger.info(f"Policy loss: {policy_loss_contrib:.6g},\tValue loss: {value_loss_contrib:.6g},\tEntropy loss: {entropy_loss_contrib:.6g}, \tlr: {self.scheduler.get_last_lr()[0]:.6g}")

        return loss_value
