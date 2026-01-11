"""
Group Relative Policy Optimization (GRPO) algorithm implementation.

Features:
- Rich console progress bars and status updates
- Detailed metric logging to files
- GPU memory monitoring
- Model architecture information display
- Generation progress tracking
"""

from dataclasses import dataclass
from typing import Callable, Optional

from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils import (
    console,
    get_training_callbacks,
    print_info,
    print_model_info,
    print_config_table,
)
from .base import (
    TrainingConfig,
    setup_logging,
    prepare_output_dir,
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

    # Print GRPO-specific config
    print_config_table(grpo_config, "GRPO Extra Config")

    # Check if test model (tiny-gpt2 or SmolLM2)
    model_name_lower = config.model_name.lower()
    is_test_model = "tiny" in model_name_lower or "smollm" in model_name_lower

    # Load model and tokenizer
    if is_test_model:
        print_info(f"Loading test model: {config.model_name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Loading model...", total=None)
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            progress.update(task, description="[cyan]Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            progress.update(task, completed=100, total=100)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            print_info("Set pad_token to eos_token")

        # Display model info
        print_model_info(model)

        model_arg = model
        processing_class = tokenizer
    else:
        print_info(f"Using model name directly: {config.model_name}")
        model_arg = config.model_name
        processing_class = None

    # Load data
    dataset = load_and_limit_dataset(
        config.dataset_name,
        config.dataset_split,
        config.max_samples,
    )

    # Determine reward function
    if grpo_config.reward_func is not None:
        reward_func = grpo_config.reward_func
        print_info("Using custom reward function")
    elif is_test_model:
        # Simple test reward function
        def reward_func(completions, **kwargs):
            return [len(c) / 100.0 for c in completions]
        print_info("Using test reward function (length-based)")
    else:
        reward_func = accuracy_reward
        print_info("Using accuracy_reward function")

    # Build trainer kwargs
    trainer_kwargs = {
        "model": model_arg,
        "reward_funcs": reward_func,
        "train_dataset": dataset,
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

    training_args_kwargs["save_steps"] = config.save_steps
    training_args_kwargs["save_strategy"] = config.save_strategy
    if config.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = config.save_total_limit

    # Get training callbacks
    callbacks = get_training_callbacks(
        output_dir=config.output_dir,
        algorithm_name="GRPO",
        verbose=config.verbose,
    )

    # Create trainer
    print_info("Creating GRPO trainer...")
    console.print(f"  [dim]num_generations:[/dim] [cyan]{grpo_config.num_generations}[/cyan]")
    console.print(f"  [dim]max_completion_length:[/dim] [cyan]{grpo_config.max_completion_length}[/cyan]")

    trainer = GRPOTrainer(
        **trainer_kwargs,
        args=GRPOConfig(**training_args_kwargs),
        callbacks=callbacks,
    )

    # Train
    console.print("[bold cyan]Starting GRPO training loop...[/bold cyan]")
    console.print("[dim]Note: GRPO generates multiple completions per sample, which may take longer.[/dim]")
    trainer.train()

    # Finalize
    finalize_training(trainer, config.output_dir, "GRPO")

    return trainer
