import torch
import torch.optim as optim
import genesis as gs
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import PolicyNetwork

class PPOAgent:
    def __init__(
         self,
         device:torch.device,
         obs_dim,
         act_dim,
         total_updates,
         lr=1e-5,
         lr_min=1e-7,
         initial_update=0,
         clip_epsilon=0.2,
         gamma=0.99,
         update_epochs=16,
         from_checkpoint=None,
    ):
        self.device = device
        self.policy = PolicyNetwork(obs_dim, act_dim, from_checkpoint=from_checkpoint).to(self.device)
        param_count = sum(p.numel() for p in self.policy.parameters())
        gs.logger.info(f"PolicyNetwork has {param_count} parameters")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.total_updates = max(1, total_updates)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.total_updates - 1),
            eta_min=lr_min,
        )
        self.completed_updates = max(0, min(initial_update, self.total_updates))
        self.current_lr = self.optimizer.param_groups[0]['lr']

    def select_action_and_get_value(self, obs):
        # obs: tensor de forma [B, obs_dim]
        if obs.device != self.device:
            obs = obs.to(self.device)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4)
        action_mean, action_std, value = self.policy.forward(obs)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob.sum(dim=-1), value

    def compute_returns(self, rewards, dones=None):
        """
        rewards: tensores de forma [T, B]
        dones: opcional, tensores [T, B] indicando el fin del episodio.
        Calcula los retornos descontados para cada entorno y, si existe,
        respeta las fronteras de episodio.
        """
        T, B = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(B, device=rewards.device)
        if dones is None:
            dones = torch.zeros_like(rewards, dtype=torch.bool, device=rewards.device)
        else:
            dones = dones.to(rewards.device)
        for t in reversed(range(T)):
            done_mask = 1.0 - dones[t].to(dtype=rewards.dtype)
            next_return = rewards[t] + self.gamma * next_return * done_mask
            steps_left = max(1, T - t)
            # Nota: dividimos por la suma de los factores de descuento para que el
            # valor dependa principalmente del estado actual y no de los pasos
            # restantes; esto no es PPO clásico y habría que revisarlo si cambiamos
            # a episodios terminados.
            if self.gamma != 1:
                weight_sum = (1 - self.gamma ** steps_left) / (1 - self.gamma)
            else:
                weight_sum = float(steps_left)
            returns[t] = next_return / weight_sum
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
        dones = rollout.get('dones')    # opcional [T, B]

        # Calcular retornos descontados para cada entorno
        returns = self.compute_returns(rewards, dones=dones)  # [T, B]

        # Aplanar tensores de [T, B, ...] a [T*B, ...]
        T, B = rewards.shape
        obs = obs.view(T * B, -1)
        actions = actions.view(T * B, -1)
        old_log_probs = old_log_probs.view(T * B)
        returns = returns.view(T * B)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1e4, neginf=-1e4)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=1e4, neginf=-1e4)

        # Calcular ventajas
        flat_values = torch.nan_to_num(values.view(-1), nan=0.0, posinf=1e4, neginf=-1e4)
        advantages = returns - flat_values
        advantages = advantages / (advantages.std() + 1e-8)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e4, neginf=-1e4)

        # Mover tensores a CUDA (si no lo están)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Hiperparámetros para balancear las pérdidas
        value_coef = 0.6     # Coeficiente para la pérdida del valor
        entropy_coef = 0.01  # Coeficiente para la bonificación de entropía

        # Update learning rate for this PPO iteration using cosine decay.
        epoch = min(self.completed_updates, self.total_updates - 1)
        self.scheduler.step(epoch)
        self.current_lr = self.scheduler.get_last_lr()[0]

        loss_value = 0.0
        for i in range(self.update_epochs):
            # Forward pass para obtener parámetros de la política y el valor
            mean, std, value_pred = self.policy(obs)  # value_pred shape: [T*B, 1]
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e4, neginf=-1e4)
            std = torch.nan_to_num(std, nan=1.0, posinf=1e4, neginf=-1e4)
            std = torch.clamp(std, min=1e-3, max=1e3)
            value_pred = torch.nan_to_num(value_pred, nan=0.0, posinf=1e4, neginf=-1e4)
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
            entropy_loss_contrib = entropy_coef * (-entropy.mean())

            # Pérdida total (con bonificación de entropía)
            loss = policy_loss_contrib + value_loss_contrib + entropy_loss_contrib

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            loss_value = loss.item()
            gs.logger.info(f"Policy loss: {policy_loss_contrib:.6g},\tValue loss: {value_loss_contrib:.6g},\tEntropy loss: {entropy_loss_contrib:.6g}, \tlr: {self.current_lr:.6g}")

        self.completed_updates += 1
        return loss_value, self.current_lr
