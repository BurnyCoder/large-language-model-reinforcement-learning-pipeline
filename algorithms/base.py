"""
Base training infrastructure shared across all algorithms.

Provides:
- TrainingConfig dataclass for common parameters
- Shared utility functions for logging, data loading, etc.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset


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
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Saving - uses built-in step-based saving
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = 3  # Keep only last N checkpoints

    # Test mode
    clean_output_dir: bool = False  # If True, removes existing output_dir before training


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging to console and file.

    Returns the logger for the calling module.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{output_dir}/training.log"),
        ],
    )
    return logging.getLogger(__name__)


def prepare_output_dir(output_dir: str, clean: bool = False) -> None:
    """
    Prepare the output directory.

    Args:
        output_dir: Path to create
        clean: If True, removes existing directory first
    """
    if clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def load_and_limit_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Load dataset and optionally limit to max_samples.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        max_samples: If set, limit dataset to this many samples

    Returns:
        The dataset
    """
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    return dataset


def finalize_training(trainer, output_dir: str, algorithm_name: str) -> None:
    """
    Standard training finalization: save model and print success.
    """
    trainer.save_model(output_dir)
    print(f"{algorithm_name} model saved to {output_dir}")
