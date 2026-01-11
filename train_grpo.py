from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
import logging
import os
from training_utils import create_training_callbacks

output_dir = "Qwen2-0.5B-GRPO"
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

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# Create callbacks for periodic saving (every 20 min) and folder size monitoring (max 20 GB)
callbacks = create_training_callbacks(
    output_dir=output_dir,
    save_interval_minutes=20,
    max_folder_size_gb=20.0,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    callbacks=callbacks,
    args=GRPOConfig(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        # Maximize GPU utilization
        per_device_train_batch_size=2,  # Reduced for 8GB VRAM (generation overhead)
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        use_liger_kernel=True,  # 60% memory reduction
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # Verbose logging
        logging_steps=1,
        logging_strategy="steps",
        log_level="info",
        report_to=["tensorboard"],
    ),
)
trainer.train()

# Save the model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
