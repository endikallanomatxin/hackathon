import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

def show_reward_info(mean_reward, loss, learning_rate, reward_dict):
    print("\n")
    print(f"REWARD {mean_reward:.6g},\tLOSS {loss:.6g},\tLR {learning_rate:.6g}")
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
        print(line)


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


class TrainingRunManager:
    """
    Handles run directory creation, checkpoint discovery, TensorBoard seeding,
    and logger instantiation so training scripts stay tidy.
    """
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def prepare_run(self, run_name: str, resume_latest: bool):
        run_dir = self.log_dir / run_name
        run_dir.mkdir()

        resume_info = self._find_latest_run(exclude_run_name=run_name) if resume_latest else None
        checkpoint_path = resume_info['model_path'] if resume_info else None

        if resume_info is not None:
            self._seed_tensorboard_history(
                resume_info['run_dir'],
                run_dir,
                resume_info['checkpoint_step'],
            )
            offset = resume_info['checkpoint_step'] + 1
        else:
            offset = 0

        logger = TensorboardLogger(self.log_dir, run_name, global_step_offset=offset)
        return os.fspath(run_dir), checkpoint_path, logger

    def _find_latest_run(self, exclude_run_name: str | None):
        run_dirs = [
            entry for entry in self.log_dir.iterdir()
            if entry.is_dir() and entry.name != exclude_run_name
        ]
        if not run_dirs:
            print("No previous run directories found. Using random initialization")
            return None
        run_dirs.sort(key=lambda entry: (entry.stat().st_mtime, entry.name))
        latest_run_dir = run_dirs[-1]
        checkpoints_dir = latest_run_dir / 'checkpoints'
        if not checkpoints_dir.exists():
            print("No checkpoints folder found. Using random initialization")
            return None
        checkpoint_dirs = sorted([p for p in checkpoints_dir.iterdir() if p.is_dir()])
        if not checkpoint_dirs:
            print("No checkpoints found. Using random initialization")
            return None
        latest_checkpoint = checkpoint_dirs[-1]
        model_path = latest_checkpoint / 'policy-model.pth'
        if not model_path.exists():
            print("No model found. Using random initialization")
            return None
        checkpoint_step = int(latest_checkpoint.name)
        print(f"Loading model from checkpoint {latest_checkpoint.name} of run {latest_run_dir.name}")
        return {
            'model_path': os.fspath(model_path),
            'checkpoint_step': checkpoint_step,
            'run_dir': os.fspath(latest_run_dir),
        }

    def _seed_tensorboard_history(self, previous_run_dir: str, new_run_dir: Path, max_step: int):
        if max_step < 0:
            return
        prev_path = Path(previous_run_dir)
        if not prev_path.is_dir():
            print(f"WARNING: Cannot seed TensorBoard history: {previous_run_dir} not found")
            return
        accumulator = event_accumulator.EventAccumulator(str(prev_path), size_guidance={
            event_accumulator.SCALARS: 0,
        })
        try:
            accumulator.Reload()
        except Exception as exc:
            print(f"WARNING: Failed to read TensorBoard data from {previous_run_dir}: {exc}")
            return
        scalar_tags = accumulator.Tags().get('scalars', [])
        if not scalar_tags:
            print(f"No scalar tags found in {previous_run_dir}; skipping history copy")
            return
        writer = SummaryWriter(log_dir=str(new_run_dir))
        try:
            for tag in scalar_tags:
                for event in accumulator.Scalars(tag):
                    if event.step <= max_step:
                        writer.add_scalar(tag, event.value, event.step)
            writer.flush()
        finally:
            writer.close()
