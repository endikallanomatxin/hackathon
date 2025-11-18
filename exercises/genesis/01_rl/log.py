import os
import shutil
import genesis as gs
from torch.utils.tensorboard import SummaryWriter

def get_latest_model(log_dir,
                     run_name:str|None=None):

    # Check if there are any runs in the log directory
    entries = os.listdir(log_dir)
    if len(entries) == 0:
        gs.logger.info("No runs found. Using random initialization")
        return None

    # Keep only directories (each directory represents a run) and sort by mtime/name
    run_dirs = []
    for entry in entries:
        if entry == run_name:
            continue
        full_path = os.path.join(log_dir, entry)
        if os.path.isdir(full_path):
            run_dirs.append((os.path.getmtime(full_path), entry))

    if len(run_dirs) == 0:
        gs.logger.info("No previous run directories found. Using random initialization")
        return None

    run_dirs.sort(key=lambda x: (x[0], x[1]))
    latest_run = run_dirs[-1][1]
    gs.logger.info(f"Latest run is: {latest_run}")

    # Look for checkpoints folder
    checkpoints_dir = os.path.join(log_dir, latest_run, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        gs.logger.info("No checkpoints folder found. Using random initialization")
        return None

    # Get the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if len(checkpoints) == 0:
        gs.logger.info("No checkpoints found. Using random initialization")
        return None
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]

    #Check for the model
    model_path = os.path.join(checkpoints_dir, latest_checkpoint, 'policy-model.pth')
    if not os.path.exists(model_path):
        gs.logger.info("No model found. Using random initialization")
        return None

    # If we are here, we have a model
    # Copy existing TensorBoard events so the new run continues the curve
    if run_name is not None:
        prev_run_dir = os.path.join(log_dir, latest_run)
        new_run_dir = os.path.join(log_dir, run_name)
        if os.path.isdir(new_run_dir):
            tb_files = [
                f for f in os.listdir(prev_run_dir)
                if f.startswith("events.out.tfevents")
            ]
            for filename in tb_files:
                src = os.path.join(prev_run_dir, filename)
                dst = os.path.join(new_run_dir, filename)
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                    gs.logger.info(f"Copied TensorBoard log {filename} to new run directory")
                except OSError as exc:
                    gs.logger.warning(f"Failed to copy TensorBoard log {filename}: {exc}")

    # TODO: Put hyperparameters in another file
    #       And also copy it between runs

    # Loading the latest model
    gs.logger.info(f"Loading model from checkpoint {latest_checkpoint} of run {latest_run}")
    return model_path

def show_reward_info(mean_reward, loss, learning_rate, reward_dict):
    gs.logger.info(f"REWARD {mean_reward:.6g},\tLOSS {loss:.6g},\tLR {learning_rate:.6g}")
    keys = list(reward_dict.keys())
    max_odd_key_len = max(len(key) for key in keys[0::2])
    max_even_key_len = max(len(key) for key in keys[1::2])
    for i in range(0, len(keys), 2):
        key1 = keys[i]
        val1 = reward_dict[key1]
        line = f"{key1:<{max_odd_key_len+1}}: {val1:>12.6g}"
        if i + 1 < len(keys):
            key2 = keys[i+1]
            val2 = reward_dict[key2]
            line += f"\t\t{key2:<{max_even_key_len+1}}: {val2:>12.6g}"
        gs.logger.info(line)


class TensorboardLogger:
    """
    Thin wrapper around SummaryWriter so training can stream metrics to TensorBoard.
    """
    def __init__(self, log_dir, run_name: str):
        base_dir = os.fspath(log_dir)
        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_dir)

    def log_update(self, step: int, mean_reward: float, loss: float, learning_rate: float, reward_dict: dict[str, float]):
        self.writer.add_scalar('train/mean_reward', mean_reward, step)
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/learning_rate', learning_rate, step)
        for key, value in reward_dict.items():
            self.writer.add_scalar(f'rewards/{key}', value, step)

    def close(self):
        self.writer.close()
