"""
Training pipeline that runs all algorithms with given configs.
"""

import time
from typing import Dict

from algorithms import TrainingConfig, train_sft, train_reward, train_dpo, train_grpo
from algorithms.grpo import GRPOExtraConfig


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
    print("=" * 70)
    print("STARTING TRAINING PIPELINE")
    print("=" * 70)
    print(f"Algorithms to run: {', '.join(configs.keys())}")
    print("=" * 70)

    start_time = time.time()
    results = []

    for algo, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Running {algo.upper()} training")
        print(f"Model: {config.model_name}")
        print(f"Output: {config.output_dir}")
        print("=" * 60)

        algo_start = time.time()

        try:
            if algo == "grpo" and grpo_extra is not None:
                train_grpo(config, grpo_extra)
            else:
                TRAINERS[algo](config)
            status = "PASSED"
        except Exception as e:
            print(f"Error in {algo}: {e}")
            status = "FAILED"

        algo_duration = time.time() - algo_start
        results.append((algo, status, algo_duration))

        if status == "PASSED":
            print(f"Completed {algo.upper()} in {algo_duration:.1f}s")
        else:
            print(f"Failed {algo.upper()} after {algo_duration:.1f}s")

    total_duration = time.time() - start_time

    # Print summary
    print(f"\n{'='*70}")
    print("PIPELINE RESULTS")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    for algo, status, duration in results:
        status_icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_icon} {algo.upper()}: {duration:.1f}s")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed")
    print(f"Total time: {total_duration:.1f}s")
    print("=" * 70)

    if failed > 0:
        raise RuntimeError(f"{failed} algorithm(s) failed")
