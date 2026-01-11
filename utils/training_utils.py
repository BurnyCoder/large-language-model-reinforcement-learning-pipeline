"""
Training utilities for periodic checkpointing and folder size management.

Supports both production (minutes) and test (seconds) configurations.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint callbacks."""

    save_interval_seconds: float
    max_folder_size_gb: float = 20.0
    folder_check_interval_steps: int = 100
    auto_skip_size_check: bool = False

    @classmethod
    def production(cls, save_interval_minutes: int = 20) -> "CheckpointConfig":
        """Create production config with minute-based intervals."""
        return cls(
            save_interval_seconds=save_interval_minutes * 60,
            max_folder_size_gb=20.0,
            folder_check_interval_steps=100,
            auto_skip_size_check=False,
        )

    @classmethod
    def test(cls, save_interval_seconds: int = 30) -> "CheckpointConfig":
        """Create test config with second-based intervals."""
        return cls(
            save_interval_seconds=save_interval_seconds,
            max_folder_size_gb=1.0,
            folder_check_interval_steps=5,
            auto_skip_size_check=True,
        )


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
    return size_bytes / (1024**3)


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


class TimedCheckpointCallback(TrainerCallback):
    """Callback that saves checkpoints at regular time intervals."""

    def __init__(self, save_interval_seconds: float, output_dir: Optional[str] = None):
        self.save_interval_seconds = save_interval_seconds
        self.output_dir = output_dir
        self.last_save_time = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.last_save_time = time.time()
        if self.save_interval_seconds >= 60:
            interval_str = f"{self.save_interval_seconds / 60:.0f} minutes"
        else:
            interval_str = f"{self.save_interval_seconds:.0f} seconds"
        logger.info(f"TimedCheckpointCallback: Will save checkpoints every {interval_str}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        current_time = time.time()
        elapsed = current_time - self.last_save_time

        if elapsed >= self.save_interval_seconds:
            save_dir = self.output_dir or args.output_dir
            checkpoint_dir = os.path.join(save_dir, f"checkpoint-time-{int(current_time)}")

            logger.info(
                f"TimedCheckpointCallback: {elapsed:.1f}s elapsed, triggering checkpoint save"
            )

            control.should_save = True
            self.last_save_time = current_time

        return control


class FolderSizeCheckCallback(TrainerCallback):
    """Callback that monitors folder size and prompts user when limit is exceeded."""

    def __init__(
        self,
        folder_path: str,
        max_size_gb: float = 20.0,
        check_interval_steps: int = 100,
        auto_skip: bool = False,
    ):
        self.folder_path = Path(folder_path)
        self.max_size_bytes = max_size_gb * (1024**3)
        self.max_size_gb = max_size_gb
        self.check_interval_steps = check_interval_steps
        self.last_check_step = 0
        self.auto_skip = auto_skip

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info(
            f"FolderSizeCheckCallback: Monitoring '{self.folder_path}' "
            f"with {self.max_size_gb} GB limit (auto_skip={self.auto_skip})"
        )
        self._check_and_prompt(state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step - self.last_check_step >= self.check_interval_steps:
            self._check_and_prompt(state)
            self.last_check_step = state.global_step

        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._check_and_prompt(state)
        return control

    def _check_and_prompt(self, state: TrainerState):
        """Check folder size and prompt user if limit exceeded."""
        current_size = get_folder_size_bytes(self.folder_path)

        logger.info(
            f"FolderSizeCheckCallback: Current size of '{self.folder_path}': "
            f"{format_size(current_size)}"
        )

        if current_size > self.max_size_bytes:
            if self.auto_skip:
                logger.warning(
                    f"Folder size {format_size(current_size)} exceeds "
                    f"{self.max_size_gb} GB limit - auto-skipping for test"
                )
            else:
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
            user_input = input(
                "\nPress ENTER after cleanup to check size again "
                "(or type 'skip' to continue anyway): "
            )

            if user_input.strip().lower() == "skip":
                logger.warning("User chose to skip folder size check")
                print("Continuing training despite folder size limit...")
                break

            new_size = get_folder_size_bytes(self.folder_path)
            new_size_gb = bytes_to_gb(new_size)

            if new_size <= self.max_size_bytes:
                print(
                    f"Folder size is now {format_size(new_size)} - below limit. "
                    "Resuming training..."
                )
                logger.info(
                    f"Folder size reduced to {format_size(new_size)}, resuming training"
                )
                break
            else:
                print(f"Folder size is still {format_size(new_size)} ({new_size_gb:.2f} GB)")
                print(f"Please reduce it to below {self.max_size_gb} GB to continue.")


def create_training_callbacks(
    output_dir: str,
    config: Optional[CheckpointConfig] = None,
    models_folder: Optional[str] = None,
) -> list:
    """
    Create training callbacks for checkpointing and size management.

    Args:
        output_dir: The output directory for the current training run
        config: CheckpointConfig instance. Defaults to production config.
        models_folder: Folder to monitor for size. If None, uses cwd.

    Returns:
        List of callbacks to pass to trainer

    Example (production):
        callbacks = create_training_callbacks(
            output_dir="Qwen2.5-0.5B-SFT",
            config=CheckpointConfig.production(save_interval_minutes=20),
        )

    Example (test):
        callbacks = create_training_callbacks(
            output_dir="test-model-sft",
            config=CheckpointConfig.test(save_interval_seconds=10),
        )
    """
    if config is None:
        config = CheckpointConfig.production()

    if models_folder is None:
        models_folder = Path.cwd()

    return [
        TimedCheckpointCallback(
            save_interval_seconds=config.save_interval_seconds,
            output_dir=output_dir,
        ),
        FolderSizeCheckCallback(
            folder_path=models_folder,
            max_size_gb=config.max_folder_size_gb,
            check_interval_steps=config.folder_check_interval_steps,
            auto_skip=config.auto_skip_size_check,
        ),
    ]
