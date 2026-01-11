"""
Run all training pipelines.

Usage:
    python run_all.py              # Run both test (tiny_gpt2) and production (qwen2.5_0.5)
    python run_all.py --test       # Run only test pipeline (fast)
    python run_all.py --prod       # Run only production pipeline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM RL training pipelines")
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run only test pipeline (tiny_gpt2)",
    )
    parser.add_argument(
        "--prod",
        "-p",
        action="store_true",
        help="Run only production pipeline (qwen2.5_0.5)",
    )
    return parser.parse_args()


def run_script(script_name: str, script_dir: Path) -> tuple[str, int, float]:
    """Run a script and return (name, return_code, duration)."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_dir / script_name],
        cwd=script_dir,
    )
    duration = time.time() - start

    return script_name, result.returncode, duration


def main():
    args = parse_args()
    script_dir = Path(__file__).parent

    # Determine which pipelines to run
    if args.test and args.prod:
        print("Error: Cannot specify both --test and --prod")
        sys.exit(1)
    elif args.test:
        scripts = ["tiny_gpt2.py"]
        mode = "TEST"
    elif args.prod:
        scripts = ["qwen2.5_0.5.py"]
        mode = "PRODUCTION"
    else:
        # Run both: test first (fast), then production
        scripts = ["tiny_gpt2.py", "qwen2.5_0.5.py"]
        mode = "ALL"

    print("=" * 70)
    print(f"RUNNING {mode} TRAINING PIPELINE(S)")
    print("=" * 70)
    print(f"Scripts to run: {', '.join(scripts)}")
    print("=" * 70)

    start_time = time.time()
    results = []

    for script in scripts:
        name, returncode, duration = run_script(script, script_dir)
        status = "PASSED" if returncode == 0 else "FAILED"
        results.append((name, status, duration))

        if returncode != 0:
            print(f"Error: {script} failed with return code {returncode}")

    total_duration = time.time() - start_time

    # Print summary
    print(f"\n{'='*70}")
    print(f"{mode} PIPELINE RESULTS")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    for script, status, duration in results:
        status_icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_icon} {script}: {duration:.1f}s")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed")
    print(f"Total time: {total_duration:.1f}s")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print(f"\nAll {mode.lower()} pipelines completed successfully!")


if __name__ == "__main__":
    main()
