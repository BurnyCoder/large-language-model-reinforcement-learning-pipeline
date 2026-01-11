"""
Supervised Fine-Tuning (SFT) algorithm implementation.
"""

from trl import SFTTrainer, SFTConfig

from .base import (
    TrainingConfig,
    setup_logging,
    prepare_output_dir,
    load_and_limit_dataset,
    finalize_training,
)


def train_sft(config: TrainingConfig) -> SFTTrainer:
    """
    Run SFT training with the given configuration.

    Args:
        config: TrainingConfig with model, dataset, and training parameters

    Returns:
        The trained SFTTrainer instance
    """
    # Setup
    prepare_output_dir(config.output_dir, clean=config.clean_output_dir)
    logger = setup_logging(config.output_dir)
    logger.info(f"Starting SFT training with model={config.model_name}")

    # Load data
    dataset = load_and_limit_dataset(
        config.dataset_name,
        config.dataset_split,
        config.max_samples,
    )
    logger.info(f"Using {len(dataset)} samples from {config.dataset_name}")

    # Build training args
    training_args_kwargs = {
        "output_dir": config.output_dir,
        "logging_dir": f"{config.output_dir}/logs",
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16,
        "use_liger_kernel": config.use_liger_kernel,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "dataloader_num_workers": config.dataloader_num_workers,
        "logging_steps": config.logging_steps,
        "logging_strategy": config.logging_strategy,
        "log_level": config.log_level,
        "report_to": config.report_to,
    }

    if config.max_steps > 0:
        training_args_kwargs["max_steps"] = config.max_steps

    training_args_kwargs["save_steps"] = config.save_steps
    training_args_kwargs["save_strategy"] = config.save_strategy
    if config.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = config.save_total_limit

    # Create trainer
    trainer = SFTTrainer(
        model=config.model_name,
        train_dataset=dataset,
        args=SFTConfig(**training_args_kwargs),
    )

    # Train
    trainer.train()

    # Finalize
    finalize_training(trainer, config.output_dir, "SFT")

    return trainer
