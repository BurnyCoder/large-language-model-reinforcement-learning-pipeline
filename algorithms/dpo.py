"""
Direct Preference Optimization (DPO) algorithm implementation.

Features:
- Rich console progress bars and status updates
- Detailed metric logging to files
- GPU memory monitoring
- Model architecture information display
"""

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils import (
    console,
    get_training_callbacks,
    print_info,
    print_model_info,
)
from .base import (
    TrainingConfig,
    setup_logging,
    prepare_output_dir,
    load_and_limit_dataset,
    finalize_training,
)


def train_dpo(config: TrainingConfig) -> DPOTrainer:
    """
    Run DPO training with the given configuration.

    DPO always requires explicit model and tokenizer loading.

    Args:
        config: TrainingConfig with model, dataset, and training parameters

    Returns:
        The trained DPOTrainer instance
    """
    # Setup
    prepare_output_dir(config.output_dir, clean=config.clean_output_dir)
    logger = setup_logging(config.output_dir)
    logger.info(f"Starting DPO training with model={config.model_name}")

    # Load model and tokenizer with progress
    print_info(f"Loading model: {config.model_name}")

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

    # Load data
    dataset = load_and_limit_dataset(
        config.dataset_name,
        config.dataset_split,
        config.max_samples,
    )

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
        "dataset_num_proc": config.dataset_num_proc,
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
        algorithm_name="DPO",
        verbose=config.verbose,
    )

    # Create trainer
    print_info("Creating DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(**training_args_kwargs),
        callbacks=callbacks,
    )

    # Train
    console.print("[bold cyan]Starting DPO training loop...[/bold cyan]")
    trainer.train()

    # Finalize
    finalize_training(trainer, config.output_dir, "DPO")

    return trainer
