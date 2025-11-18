import os
import genesis as gs
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

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

    checkpoint_step = int(latest_checkpoint)
    gs.logger.info(f"Loading model from checkpoint {latest_checkpoint} of run {latest_run}")
    return {
        'model_path': model_path,
        'checkpoint_step': checkpoint_step,
        'previous_run_dir': os.path.join(log_dir, latest_run),
    }

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
    def __init__(self, log_dir, run_name: str, global_step_offset: int = 0):
        base_dir = os.fspath(log_dir)
        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_dir)
        self.global_step_offset = global_step_offset

    def log_update(self, step: int, mean_reward: float, loss: float, learning_rate: float, reward_dict: dict[str, float]):
        global_step = step + self.global_step_offset
        self.writer.add_scalar('train/mean_reward', mean_reward, global_step)
        self.writer.add_scalar('train/loss', loss, global_step)
        self.writer.add_scalar('train/learning_rate', learning_rate, global_step)
        for key, value in reward_dict.items():
            self.writer.add_scalar(f'rewards/{key}', value, global_step)

    def close(self):
        self.writer.close()


def seed_tensorboard_history(previous_run_dir: str, new_run_dir: str, max_step: int):
    """
    Copy existing TensorBoard scalars from a previous run into the new run directory,
    but only up to the specified max_step so resumed training reflects the checkpoint.
    """
    if max_step is None or max_step < 0:
        return
    if not os.path.isdir(previous_run_dir):
        gs.logger.warning(f"Cannot seed TensorBoard history: {previous_run_dir} not found")
        return
    event_files = [
        f for f in os.listdir(previous_run_dir)
        if f.startswith("events.out.tfevents")
    ]
    if not event_files:
        gs.logger.info(f"No TensorBoard event files found in {previous_run_dir}; skipping history copy")
        return
    accumulator = event_accumulator.EventAccumulator(previous_run_dir, size_guidance={
        event_accumulator.SCALARS: 0,
    })
    try:
        accumulator.Reload()
    except Exception as exc:
        gs.logger.warning(f"Failed to read TensorBoard data from {previous_run_dir}: {exc}")
        return
    scalar_tags = accumulator.Tags().get('scalars', [])
    if not scalar_tags:
        gs.logger.info(f"No scalar tags found in {previous_run_dir}; skipping history copy")
        return
    writer = SummaryWriter(log_dir=new_run_dir)
    try:
        for tag in scalar_tags:
            for event in accumulator.Scalars(tag):
                if event.step <= max_step:
                    writer.add_scalar(tag, event.value, event.step)
        writer.flush()
    finally:
        writer.close()
