"""
Base training infrastructure shared across all algorithms.

Provides:
- TrainingConfig dataclass for common parameters
- Shared utility functions for logging, data loading, etc.
- Rich console output and progress tracking
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from utils import (
    console,
    setup_rich_logging,
    print_info,
    print_success,
    print_warning,
    print_dataset_info,
    print_model_info,
    get_training_callbacks,
    format_duration,
)


@dataclass
class TrainingConfig:
    """
    Common training configuration shared across algorithms.

    Model-specific values (model_name, output_dir, dataset_name) are passed
    as arguments - no hardcoded values here.
    """

    # Model and output - required, no defaults
    model_name: str
    output_dir: str
    dataset_name: str

    # Dataset settings
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None = use full dataset

    # Training hyperparameters
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = -1  # -1 = train on full dataset

    # Optimization flags
    gradient_checkpointing: bool = True
    bf16: bool = True
    use_liger_kernel: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4

    # Logging
    logging_steps: int = 1
    logging_strategy: str = "steps"
    log_level: str = "info"
    report_to: List[str] = field(default_factory=lambda: ["wandb", "neptune", "tensorboard"])

    # Saving - uses built-in step-based saving
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = 3  # Keep only last N checkpoints

    # Test mode
    clean_output_dir: bool = False  # If True, removes existing output_dir before training

    # Verbose logging
    verbose: bool = True  # If True, prints detailed progress information


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging with Rich handler for beautiful terminal output.

    Returns the logger for the calling module.
    """
    return setup_rich_logging(output_dir)


def prepare_output_dir(output_dir: str, clean: bool = False) -> None:
    """
    Prepare the output directory.

    Args:
        output_dir: Path to create
        clean: If True, removes existing directory first
    """
    if clean and os.path.exists(output_dir):
        print_warning(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print_info(f"Output directory ready: {output_dir}")


def load_and_limit_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Load dataset and optionally limit to max_samples.

    Displays progress and dataset information.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        max_samples: If set, limit dataset to this many samples

    Returns:
        The dataset
    """
    console.print(f"[dim]Loading dataset:[/dim] [cyan]{dataset_name}[/cyan] (split: {split})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Downloading {dataset_name}...", total=None)
        dataset = load_dataset(dataset_name, split=split)
        progress.update(task, completed=100, total=100)

    original_size = len(dataset)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print_info(f"Limited dataset from {original_size:,} to {max_samples:,} samples")

    # Print dataset info
    print_dataset_info(dataset, dataset_name)

    return dataset


def finalize_training(trainer, output_dir: str, algorithm_name: str) -> None:
    """
    Standard training finalization: save model and print success.
    """
    console.print()
    console.print(f"[dim]Saving {algorithm_name} model to:[/dim] [green]{output_dir}[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Saving model...", total=None)
        trainer.save_model(output_dir)
        progress.update(task, completed=100, total=100)

    print_success(f"{algorithm_name} model saved to {output_dir}")


def log_model_loaded(model, algorithm_name: str) -> None:
    """Log information about a loaded model."""
    console.print(f"[dim]Model loaded for {algorithm_name}[/dim]")
    print_model_info(model)
