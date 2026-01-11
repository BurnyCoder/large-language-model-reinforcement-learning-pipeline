"""
Qwen 2.5 0.5B training pipeline.

Runs all 4 algorithms (SFT, Reward, DPO, GRPO) with Qwen 2.5 0.5B models.
This is the production configuration optimized for 8GB VRAM GPUs.
"""

from algorithms import TrainingConfig
from pipeline import run_pipeline

# Qwen 2.5 0.5B models
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Checkpoint settings: 10 checkpoints per algorithm × 4 algorithms × ~4GB = ~160GB (under 200GB limit)
SAVE_TOTAL_LIMIT = 10

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
        save_steps=50,  # ~500 total steps, 10 checkpoints
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
        save_steps=187,  # ~1875 total steps, 10 checkpoints
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
        save_steps=750,  # ~7500 total steps, 10 checkpoints
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
    "grpo": TrainingConfig(
        model_name=INSTRUCT_MODEL,
        output_dir="Qwen2.5-0.5B-GRPO",
        dataset_name="trl-lib/DeepMath-103K",
        per_device_train_batch_size=2,  # Reduced for 8GB VRAM (generation overhead)
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_steps=1287,  # ~12875 total steps, 10 checkpoints
        save_total_limit=SAVE_TOTAL_LIMIT,
    ),
}

if __name__ == "__main__":
    run_pipeline(configs)
