"""
Training utilities for periodic checkpointing and folder size management.

Features:
- TimedCheckpointCallback: Saves model checkpoints every N minutes
- FolderSizeCheckCallback: Prompts user when model folder exceeds size limit
"""

import os
import time
import logging
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


def get_folder_size_bytes(folder_path: str | Path) -> int:
    """Calculate total size of a folder in bytes, including all subdirectories."""
    total_size = 0
    folder = Path(folder_path)

    if not folder.exists():
        return 0

    for entry in folder.rglob("*"):
        if entry.is_file():
            try:
                total_size += entry.stat().st_size
            except (OSError, PermissionError):
                pass

    return total_size


def bytes_to_gb(size_bytes: int) -> float:
    """Convert bytes to gigabytes."""
    return size_bytes / (1024 ** 3)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


class TimedCheckpointCallback(TrainerCallback):
    """
    Callback that saves checkpoints at regular time intervals.

    Args:
        save_interval_minutes: Number of minutes between checkpoint saves (default: 20)
        output_dir: Directory to save checkpoints. If None, uses trainer's output_dir
    """

    def __init__(self, save_interval_minutes: int = 20, output_dir: str | None = None):
        self.save_interval_seconds = save_interval_minutes * 60
        self.output_dir = output_dir
        self.last_save_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        self.last_save_time = time.time()
        logger.info(f"TimedCheckpointCallback: Will save checkpoints every {self.save_interval_seconds // 60} minutes")

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        current_time = time.time()
        elapsed = current_time - self.last_save_time

        if elapsed >= self.save_interval_seconds:
            save_dir = self.output_dir or args.output_dir
            checkpoint_dir = os.path.join(save_dir, f"checkpoint-time-{int(current_time)}")

            logger.info(f"TimedCheckpointCallback: {elapsed / 60:.1f} minutes elapsed, triggering checkpoint save")

            # Signal trainer to save
            control.should_save = True
            self.last_save_time = current_time

        return control


class FolderSizeCheckCallback(TrainerCallback):
    """
    Callback that monitors folder size and prompts user when limit is exceeded.

    Args:
        folder_path: Path to the folder to monitor (typically the models directory)
        max_size_gb: Maximum allowed size in gigabytes (default: 20)
        check_interval_steps: How often to check folder size in training steps (default: 100)
    """

    def __init__(self, folder_path: str, max_size_gb: float = 20.0,
                 check_interval_steps: int = 100):
        self.folder_path = Path(folder_path)
        self.max_size_bytes = max_size_gb * (1024 ** 3)
        self.max_size_gb = max_size_gb
        self.check_interval_steps = check_interval_steps
        self.last_check_step = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        logger.info(f"FolderSizeCheckCallback: Monitoring '{self.folder_path}' with {self.max_size_gb} GB limit")
        self._check_and_prompt(state)

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        if state.global_step - self.last_check_step >= self.check_interval_steps:
            self._check_and_prompt(state)
            self.last_check_step = state.global_step

        return control

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        # Always check after a save operation
        self._check_and_prompt(state)
        return control

    def _check_and_prompt(self, state: TrainerState):
        """Check folder size and prompt user if limit exceeded."""
        current_size = get_folder_size_bytes(self.folder_path)
        current_size_gb = bytes_to_gb(current_size)

        logger.info(f"FolderSizeCheckCallback: Current size of '{self.folder_path}': {format_size(current_size)}")

        if current_size > self.max_size_bytes:
            self._prompt_user_cleanup(current_size, state)

    def _prompt_user_cleanup(self, current_size: int, state: TrainerState):
        """Prompt user to clean up and wait until they do."""
        print("\n" + "=" * 70)
        print("WARNING: MODEL FOLDER SIZE LIMIT EXCEEDED!")
        print("=" * 70)
        print(f"Folder:       {self.folder_path}")
        print(f"Current size: {format_size(current_size)}")
        print(f"Limit:        {self.max_size_gb:.1f} GB")
        print(f"Training step: {state.global_step}")
        print("-" * 70)
        print("Training is PAUSED. Please clean up the folder to continue.")
        print("Delete old checkpoints or move them to another location.")
        print("=" * 70)

        while True:
            user_input = input("\nPress ENTER after cleanup to check size again (or type 'skip' to continue anyway): ")

            if user_input.strip().lower() == "skip":
                logger.warning("User chose to skip folder size check")
                print("Continuing training despite folder size limit...")
                break

            new_size = get_folder_size_bytes(self.folder_path)
            new_size_gb = bytes_to_gb(new_size)

            if new_size <= self.max_size_bytes:
                print(f"Folder size is now {format_size(new_size)} - below limit. Resuming training...")
                logger.info(f"Folder size reduced to {format_size(new_size)}, resuming training")
                break
            else:
                print(f"Folder size is still {format_size(new_size)} ({new_size_gb:.2f} GB)")
                print(f"Please reduce it to below {self.max_size_gb} GB to continue.")


def get_models_folder(output_dir: str) -> Path:
    """
    Get the parent folder containing all model outputs.

    If output_dir is 'Qwen2.5-0.5B-SFT', returns the parent directory
    that would contain all model folders.
    """
    return Path(output_dir).parent.absolute()


def create_training_callbacks(
    output_dir: str,
    save_interval_minutes: int = 20,
    models_folder: str | None = None,
    max_folder_size_gb: float = 20.0,
    folder_check_interval_steps: int = 100,
) -> list:
    """
    Create standard training callbacks for checkpointing and size management.

    Args:
        output_dir: The output directory for the current training run
        save_interval_minutes: Minutes between checkpoint saves (default: 20)
        models_folder: Folder to monitor for size. If None, uses parent of output_dir
        max_folder_size_gb: Max size in GB before prompting user (default: 20)
        folder_check_interval_steps: Steps between size checks (default: 100)

    Returns:
        List of callbacks to pass to trainer

    Example:
        callbacks = create_training_callbacks(
            output_dir="Qwen2.5-0.5B-SFT",
            save_interval_minutes=20,
            max_folder_size_gb=20.0,
        )
        trainer = SFTTrainer(..., callbacks=callbacks)
    """
    # Determine the models folder to monitor
    if models_folder is None:
        # Monitor the current working directory (where all model folders are)
        models_folder = Path.cwd()

    callbacks = [
        TimedCheckpointCallback(
            save_interval_minutes=save_interval_minutes,
            output_dir=output_dir,
        ),
        FolderSizeCheckCallback(
            folder_path=models_folder,
            max_size_gb=max_folder_size_gb,
            check_interval_steps=folder_check_interval_steps,
        ),
    ]

    return callbacks
