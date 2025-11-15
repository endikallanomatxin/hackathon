import os
import time
import pathlib
import torch
import genesis as gs

from env import Environment
from agent import PPOAgent
from log import get_latest_model, log_plot, show_reward_info, log_update

def train(rollout_steps=400,
          batch_size=1024,
          max_steps=2000,
          record=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = pathlib.Path(__file__).parent / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    training_run_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(log_dir, training_run_name))

    env = Environment(device=device,
                      batch_size=batch_size,
                      max_steps=max_steps,
                      record=record)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    agent = PPOAgent(
        device,
        obs_dim,
        act_dim,
        # Comment / uncomment this for model reuse
        # from_checkpoint=get_latest_model(log_dir, training_run_name)
    )

    os.makedirs(os.path.join(log_dir, training_run_name, 'checkpoints'))

    inference_every_n_steps = 10
    checkpoint_every_n_updates = 100
    plot_every_n_updates = 10

    for update in range(100_000):
        print("\n")
        gs.logger.info(f"UPDATE {update}")
        # Initialize environments (no_grad, we don't want to backprop through env)
        with torch.no_grad():
            obs = env.reset()

        obs_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        reward_dict_list = []

        checkpoint = update % checkpoint_every_n_updates == 0
        plot = update % plot_every_n_updates == 0

        for step in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value = agent.select_action_and_get_value(obs)

            obs_list.append(obs.clone().detach())            # shape [B, obs_dim]
            actions_list.append(action.clone().detach())     # shape [B, act_dim]
            log_probs_list.append(log_prob.clone().detach()) # shape [B]
            values_list.append(value)                        # shape [B]

            reward_sum = torch.zeros(env.batch_size, device=obs.device)
            reward_dict_sum = None

            for _ in range(inference_every_n_steps):
                with torch.no_grad():
                    next_obs, reward, reward_dict = env.step(action, record=checkpoint)
                reward_sum = reward_sum + reward
                if reward_dict_sum is None:
                    reward_dict_sum = {key: float(val) for key, val in reward_dict.items()}
                else:
                    for key in reward_dict_sum:
                        reward_dict_sum[key] += float(reward_dict[key])

            reward_mean = reward_sum / inference_every_n_steps
            rewards_list.append(reward_mean.clone().detach())  # shape [B]
            reward_dict_list.append(
                {key: reward_dict_sum[key] / inference_every_n_steps for key in reward_dict_sum}
            )

            obs = next_obs

        # Stack them into [T, B, ...]
        rollout = {
            'obs':      torch.stack(obs_list, dim=0),      # [T, B, obs_dim]
            'actions':  torch.stack(actions_list, dim=0),  # [T, B, act_dim]
            'log_probs':torch.stack(log_probs_list, dim=0),# [T, B]
            'values':   torch.stack(values_list, dim=0),   # [T, B]
            'rewards':  torch.stack(rewards_list, dim=0),  # [T, B]
        }

        loss = agent.update(rollout)
        mean_reward = rollout['rewards'].mean().item()

        reward_dict_mean = {}
        for key in reward_dict_list[0]:
            reward_dict_mean[key] = sum(reward_dict[key] for reward_dict in reward_dict_list) / len(reward_dict_list)

        show_reward_info(mean_reward, loss, reward_dict_mean)
        log_update(log_dir, training_run_name, update, mean_reward, loss, reward_dict_mean)

        if checkpoint:
            # Create new checkpoints folder
            checkpoint_dir = os.path.join(log_dir, training_run_name, 'checkpoints', f'{update:08d}')
            os.makedirs(checkpoint_dir)
            # Save the video if recording
            if record:
                env.save_video(os.path.join(checkpoint_dir, f"video.mp4"))
            # Save the model
            torch.save(agent.policy.state_dict(), os.path.join(checkpoint_dir, 'policy-model.pth'))

        if plot:
            # Create or update the log plot
            log_plot(log_dir, training_run_name)


if __name__ == "__main__":
    train()
