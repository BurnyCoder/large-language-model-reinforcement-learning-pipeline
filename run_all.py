import subprocess
import sys
from pathlib import Path

scripts = [
    "train_sft.py",      # ~16K samples, fastest
    "train_reward.py",   # ~60K samples, batch 32
    "train_dpo.py",      # ~60K samples, batch 16, loads 2 models
    "train_grpo.py",     # ~103K samples, generation overhead, slowest
]

script_dir = Path(__file__).parent

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, script_dir / script],
        cwd=script_dir,
    )

    if result.returncode != 0:
        print(f"Error: {script} failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"Completed {script}")

print(f"\n{'='*60}")
print("All training scripts completed successfully!")
print('='*60)
