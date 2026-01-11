"""
Training pipeline that runs all algorithms with given configs.

Features:
- Rich console output with progress bars and status indicators
- Comprehensive logging to console and files
- System information display (GPU, memory, etc.)
- Beautiful result tables
- Unique run IDs to prevent output overwrites
"""

import cache_config  # noqa: F401 - Configure HF cache before imports

import os
import time
from typing import Dict, List, Tuple

from algorithms import TrainingConfig, train_sft, train_reward, train_dpo, train_grpo
from algorithms.grpo import GRPOExtraConfig
from utils import (
    console,
    create_run_directory,
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_config_table,
    print_system_info,
    print_training_summary,
    create_pipeline_progress,
    print_pipeline_results,
    format_duration,
    save_run_info,
)


TRAINERS = {
    "sft": train_sft,
    "reward": train_reward,
    "dpo": train_dpo,
    "grpo": train_grpo,
}


def run_pipeline(
    configs: Dict[str, TrainingConfig],
    grpo_extra: GRPOExtraConfig | None = None,
) -> None:
    """
    Run training algorithms with given configs.

    Args:
        configs: Dict mapping algorithm name to its config
                 e.g., {"sft": sft_config, "reward": reward_config, ...}
        grpo_extra: Optional extra config for GRPO (reward func, generation params)
    """
    # Generate unique run ID and create run directory
    run_id, run_dir = create_run_directory()

    # Prefix all output_dirs with run directory
    for config in configs.values():
        config.output_dir = os.path.join(run_dir, config.output_dir)

    # Save run metadata
    save_run_info(run_dir, run_id, configs)

    # Print header
    print_header(
        "Training Pipeline",
        f"Run ID: {run_id} | Running {len(configs)} algorithm(s): {', '.join(configs.keys())}"
    )

    # Print system info
    print_section("System Information")
    print_system_info()

    # Print configuration overview
    print_section("Pipeline Configuration")
    for algo, config in configs.items():
        console.print(f"[bold cyan]{algo.upper()}[/bold cyan]")
        print_config_table(config, f"{algo.upper()} Config")

    if grpo_extra is not None:
        print_config_table(grpo_extra, "GRPO Extra Config")

    # Run training with progress tracking
    print_section("Training Progress")

    start_time = time.time()
    results: List[Tuple[str, str, float]] = []

    # Create pipeline progress bar
    with create_pipeline_progress() as progress:
        pipeline_task = progress.add_task(
            "[cyan]Pipeline Progress",
            total=len(configs)
        )

        for algo, config in configs.items():
            # Update progress description
            progress.update(
                pipeline_task,
                description=f"[cyan]Running {algo.upper()}..."
            )

            console.print()
            print_section(f"{algo.upper()} Training")
            print_info(f"Model: {config.model_name}")
            print_info(f"Dataset: {config.dataset_name}")
            print_info(f"Output: {config.output_dir}")

            algo_start = time.time()

            try:
                if algo == "grpo" and grpo_extra is not None:
                    train_grpo(config, grpo_extra)
                else:
                    TRAINERS[algo](config)
                status = "PASSED"
                print_success(f"{algo.upper()} completed in {format_duration(time.time() - algo_start)}")
            except Exception as e:
                console.print_exception()
                print_error(f"{algo.upper()} failed: {e}")
                status = "FAILED"

            algo_duration = time.time() - algo_start
            results.append((algo, status, algo_duration))

            # Print algorithm summary
            print_training_summary(
                algo,
                algo_duration,
                status="completed" if status == "PASSED" else "failed"
            )

            # Update pipeline progress
            progress.advance(pipeline_task)

    total_duration = time.time() - start_time

    # Print final results
    print_section("Pipeline Results")
    print_pipeline_results(results)

    # Summary stats
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    console.print()
    if failed == 0:
        print_success(f"All {passed} algorithm(s) completed successfully!")
    else:
        print_error(f"{failed} algorithm(s) failed, {passed} passed")

    console.print(f"[dim]Total pipeline time: {format_duration(total_duration)}[/dim]")
    console.print()

    if failed > 0:
        raise RuntimeError(f"{failed} algorithm(s) failed")
