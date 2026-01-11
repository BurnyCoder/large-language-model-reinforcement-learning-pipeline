"""
Test runner that executes all test training scripts.

Runs minimal versions of all training scripts to validate:
- Model loading and training loop
- Dataset loading and processing
- Checkpoint saving (every 10 seconds)
- Folder size monitoring

Expected total runtime: ~1-2 minutes
"""

import subprocess
import sys
import time
from pathlib import Path

scripts = [
    "test_train_sft.py",     # Test SFT training
    "test_train_reward.py",  # Test Reward model training
    "test_train_dpo.py",     # Test DPO training
    "test_train_grpo.py",    # Test GRPO training
]

script_dir = Path(__file__).parent

print("=" * 70)
print("RUNNING TEST TRAINING SUITE")
print("=" * 70)
print("This will run minimal versions of all training scripts to validate")
print("the training pipeline works correctly.")
print(f"Scripts to run: {len(scripts)}")
print("=" * 70)

start_time = time.time()
results = []

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)

    script_start = time.time()
    result = subprocess.run(
        [sys.executable, script_dir / script],
        cwd=script_dir,
    )
    script_duration = time.time() - script_start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    results.append((script, status, script_duration))

    if result.returncode != 0:
        print(f"Error: {script} failed with return code {result.returncode}")
        print(f"Continuing with remaining tests...")
    else:
        print(f"Completed {script} in {script_duration:.1f}s")

total_duration = time.time() - start_time

# Print summary
print(f"\n{'='*70}")
print("TEST RESULTS SUMMARY")
print('='*70)

passed = sum(1 for _, status, _ in results if status == "PASSED")
failed = sum(1 for _, status, _ in results if status == "FAILED")

for script, status, duration in results:
    status_icon = "✓" if status == "PASSED" else "✗"
    print(f"  {status_icon} {script}: {status} ({duration:.1f}s)")

print('-'*70)
print(f"Total: {passed} passed, {failed} failed")
print(f"Total time: {total_duration:.1f}s")
print('='*70)

if failed > 0:
    sys.exit(1)
else:
    print("\nAll test training scripts completed successfully!")
    sys.exit(0)
