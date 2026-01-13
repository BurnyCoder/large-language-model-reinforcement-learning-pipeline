"""
Qwen 2.5 0.5B training pipeline.

Runs all 4 algorithms (SFT, Reward, DPO, GRPO) with Qwen 2.5 0.5B models.
This is the production configuration optimized for 6GB+ VRAM GPUs.

Usage:
    python qwen2.5_0.5b.py              # Run all 4 stages
    python qwen2.5_0.5b.py --stage grpo # Run only GRPO stage
"""

import argparse

import cache_config  # noqa: F401 - Configure HF cache before imports

from algorithms import TrainingConfig
from pipeline import run_pipeline

# Qwen 2.5 0.5B models
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Checkpoint settings: 12 checkpoints per algorithm × 4 × ~4GB = ~192GB (under 200GB)
SAVE_TOTAL_LIMIT = 12

configs = {
    "sft": TrainingConfig(
        model_name=BASE_MODEL,
        output_dir="Qwen2.5-0.5B-SFT",
        dataset_name="trl-lib/Capybara",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_steps=50,  
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
    "reward": TrainingConfig(
        model_name=INSTRUCT_MODEL,
        output_dir="Qwen2.5-0.5B-Reward",
        dataset_name="trl-lib/ultrafeedback_binarized",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_steps=50, 
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
    "dpo": TrainingConfig(
        model_name=INSTRUCT_MODEL,
        output_dir="Qwen2.5-0.5B-DPO",
        dataset_name="trl-lib/ultrafeedback_binarized",
        per_device_train_batch_size=2,  # Reduced for 8GB VRAM (loads 2 models)
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_steps=50,  
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
    "grpo": TrainingConfig(
        model_name=INSTRUCT_MODEL,
        output_dir="Qwen2.5-0.5B-GRPO",
        dataset_name="trl-lib/DeepMath-103K",
        per_device_train_batch_size=2,  # Liger kernel enables batch=2 with 2048 tokens
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,  # ~60% memory reduction on training ops
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_steps=120,
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen 2.5 0.5B training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        "-s",
        choices=["sft", "reward", "dpo", "grpo"],
        help="Run only a specific stage (default: run all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage:
        filtered_configs = {args.stage: configs[args.stage]}
        run_pipeline(filtered_configs)
    else:
        run_pipeline(configs)
