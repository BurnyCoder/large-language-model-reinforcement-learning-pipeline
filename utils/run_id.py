"""
Utilities for generating unique run IDs and managing run directories.
"""

import json
import os
import random
import string
from datetime import datetime


def generate_run_id() -> str:
    """Generate unique run ID: YYYYMMDD_HHMMSS_xxxx"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{suffix}"


def create_run_directory(base_dir: str = "runs") -> tuple[str, str]:
    """Create run directory, return (run_id, run_dir_path)."""
    run_id = generate_run_id()
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def save_run_info(run_dir: str, run_id: str, configs: dict) -> None:
    """Save run metadata to run_info.json."""
    info = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "algorithms": list(configs.keys()),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
