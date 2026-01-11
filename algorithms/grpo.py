"""
Group Relative Policy Optimization (GRPO) algorithm implementation.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import (
    TrainingConfig,
    setup_logging,
    prepare_output_dir,
    get_callbacks,
    load_and_limit_dataset,
    finalize_training,
)


@dataclass
class GRPOExtraConfig:
    """GRPO-specific configuration beyond the base TrainingConfig."""

    reward_func: Optional[Callable] = None  # Will default to accuracy_reward
    num_generations: int = 4
    max_completion_length: int = 256


def train_grpo(
    config: TrainingConfig,
    grpo_config: Optional[GRPOExtraConfig] = None,
) -> GRPOTrainer:
    """
    Run GRPO training with the given configuration.

    Args:
        config: Base TrainingConfig
        grpo_config: GRPO-specific settings (reward function, generation params)

    Returns:
        The trained GRPOTrainer instance
    """
    if grpo_config is None:
        grpo_config = GRPOExtraConfig()

    # Setup
    prepare_output_dir(config.output_dir, clean=config.clean_output_dir)
    logger = setup_logging(config.output_dir)
    logger.info(f"Starting GRPO training with model={config.model_name}")

    # Check if test model
    is_test_model = "tiny" in config.model_name.lower()

    # Load model and tokenizer
    if is_test_model:
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        model_arg = model
        processing_class = tokenizer
    else:
        model_arg = config.model_name
        processing_class = None

    # Load data
    dataset = load_and_limit_dataset(
        config.dataset_name,
        config.dataset_split,
        config.max_samples,
    )
    logger.info(f"Using {len(dataset)} samples from {config.dataset_name}")

    # Determine reward function
    if grpo_config.reward_func is not None:
        reward_func = grpo_config.reward_func
    elif is_test_model:
        # Simple test reward function
        def reward_func(completions, **kwargs):
            return [len(c) / 100.0 for c in completions]
    else:
        reward_func = accuracy_reward

    # Create callbacks
    callbacks = get_callbacks(config)

    # Build trainer kwargs
    trainer_kwargs = {
        "model": model_arg,
        "reward_funcs": reward_func,
        "train_dataset": dataset,
        "callbacks": callbacks,
    }
    if processing_class is not None:
        trainer_kwargs["processing_class"] = processing_class

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
        "num_generations": grpo_config.num_generations,
        "max_completion_length": grpo_config.max_completion_length,
        "logging_steps": config.logging_steps,
        "logging_strategy": config.logging_strategy,
        "log_level": config.log_level,
        "report_to": config.report_to,
    }

    if config.max_steps > 0:
        training_args_kwargs["max_steps"] = config.max_steps

    if config.save_steps is not None:
        training_args_kwargs["save_steps"] = config.save_steps
        training_args_kwargs["save_strategy"] = "steps"

    # Create trainer
    trainer = GRPOTrainer(
        **trainer_kwargs,
        args=GRPOConfig(**training_args_kwargs),
    )

    # Train
    trainer.train()

    # Finalize
    finalize_training(trainer, config.output_dir, "GRPO")

    return trainer
