"""
Test version of SFT training with minimal configuration.

Uses:
- Smallest possible model (tiny GPT-2 ~17M params)
- Minimal dataset (10 samples)
- Minimal training steps (10 steps)
- Fast checkpoint saving (every 10 seconds)
"""

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import logging
import os
import shutil
from test_training_utils import create_test_training_callbacks

# Test configuration
TEST_MODEL = "sshleifer/tiny-gpt2"  # ~17M parameters, very fast
TEST_MAX_STEPS = 10
TEST_DATASET_SIZE = 10
TEST_SAVE_INTERVAL_SECONDS = 10

output_dir = "test-model-sft"

# Clean up previous test run
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Enable verbose logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{output_dir}/training.log"),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Starting test SFT training with model={TEST_MODEL}, max_steps={TEST_MAX_STEPS}, dataset_size={TEST_DATASET_SIZE}")

# Load minimal dataset
dataset = load_dataset("trl-lib/Capybara", split="train")
dataset = dataset.select(range(min(TEST_DATASET_SIZE, len(dataset))))
logger.info(f"Using {len(dataset)} samples from dataset")

# Create callbacks with fast checkpoint saving
callbacks = create_test_training_callbacks(
    output_dir=output_dir,
    save_interval_seconds=TEST_SAVE_INTERVAL_SECONDS,
    max_folder_size_gb=1.0,
)

trainer = SFTTrainer(
    model=TEST_MODEL,
    train_dataset=dataset,
    callbacks=callbacks,
    args=SFTConfig(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        max_steps=TEST_MAX_STEPS,
        # Minimal batch size for fast testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        # Disable heavy optimizations for speed
        gradient_checkpointing=False,
        bf16=False,
        use_liger_kernel=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Verbose logging every step
        logging_steps=1,
        logging_strategy="steps",
        log_level="info",
        report_to=["tensorboard"],
        # Save checkpoint every 5 steps to test saving
        save_steps=5,
        save_strategy="steps",
    ),
)
trainer.train()

# Save the model
trainer.save_model(output_dir)
print(f"Test SFT model saved to {output_dir}")
