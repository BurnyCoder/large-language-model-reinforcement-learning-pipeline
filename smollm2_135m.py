"""
Test training pipeline with SmolLM2-135M models.

Model selection per TRL patterns:
- SFT: Instruct (SmolLM2 base lacks chat template, unlike Qwen base)
- Reward: Base (adds classification head)
- DPO/GRPO: Instruct

This is the test configuration for fast validation of the training pipeline.
"""

import cache_config  # noqa: F401 - Configure HF cache before imports

from algorithms import TrainingConfig
from algorithms.grpo import GRPOExtraConfig
from pipeline import run_pipeline

# SmolLM2 models - smallest with proper architecture for all TRL algorithms
MODEL_BASE = "HuggingFaceTB/SmolLM2-135M"
MODEL_INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Test training configuration
TEST_MAX_STEPS = 10
TEST_MAX_SAMPLES = 10

# Common test settings
TEST_SETTINGS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": TEST_MAX_STEPS,
    "max_samples": TEST_MAX_SAMPLES,
    "gradient_checkpointing": False,
    "bf16": False,
    "use_liger_kernel": False,
    "dataloader_pin_memory": False,
    "dataloader_num_workers": 0,
    "clean_output_dir": True,
    "save_steps": 5,
    "save_strategy": "steps",
    "save_total_limit": 2,
}

configs = {
    "sft": TrainingConfig(
        model_name=MODEL_INSTRUCT,  # SmolLM2 base lacks chat template
        output_dir="test-model-sft",
        dataset_name="trl-lib/Capybara",
        **TEST_SETTINGS,
    ),
    "reward": TrainingConfig(
        model_name=MODEL_BASE,  # Reward adds classification head
        output_dir="test-model-reward",
        dataset_name="trl-lib/ultrafeedback_binarized",
        **TEST_SETTINGS,
    ),
    "dpo": TrainingConfig(
        model_name=MODEL_INSTRUCT,  # DPO uses instruct model
        output_dir="test-model-dpo",
        dataset_name="trl-lib/ultrafeedback_binarized",
        **TEST_SETTINGS,
    ),
    "grpo": TrainingConfig(
        model_name=MODEL_INSTRUCT,  # GRPO uses instruct model
        output_dir="test-model-grpo",
        dataset_name="trl-lib/DeepMath-103K",
        **TEST_SETTINGS,
    ),
}

# GRPO extra config with minimal generation settings
grpo_extra = GRPOExtraConfig(
    num_generations=2,
    max_completion_length=32,
)

if __name__ == "__main__":
    run_pipeline(configs, grpo_extra=grpo_extra)
