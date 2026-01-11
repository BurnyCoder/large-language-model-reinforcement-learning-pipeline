# LLM RL - Language Model Reinforcement Learning Pipeline

A complete, memory-efficient pipeline for training language models using reinforcement learning techniques. Built on [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) from Hugging Face.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![TRL](https://img.shields.io/badge/TRL-0.26.0+-green.svg)

## Overview

This project demonstrates a complete LLM training pipeline with four key training methods:

- **Supervised Fine-Tuning (SFT)** - Train models on high-quality instruction data
- **Reward Modeling** - Train models to evaluate response quality
- **Direct Preference Optimization (DPO)** - Align models with human preferences
- **Group Relative Policy Optimization (GRPO)** - Memory-efficient reinforcement learning

All scripts are optimized for **6GB VRAM GPUs** using memory-efficient techniques like Liger Kernel and gradient checkpointing.

## Training Pipeline

```
+---------------------------------------------------------------------+
|                     LLM RL Training Pipeline                        |
+---------------------------------------------------------------------+
|                                                                     |
|  +----------+    +-----------+    +----------+    +-----------+     |
|  |   SFT    |--->|  Reward   |--->|   DPO    |--->|   GRPO    |     |
|  | (16K)    |    |  (60K)    |    |  (60K)   |    |  (103K)   |     |
|  +----------+    +-----------+    +----------+    +-----------+     |
|                                                                     |
|  Instruction     Response         Preference      Reinforcement     |
|  Following       Quality          Alignment       Learning          |
|                                                                     |
+---------------------------------------------------------------------+
```

### Training Stages

1. **SFT (Supervised Fine-Tuning)**: Teaches the model to follow instructions using the Capybara dataset (~16K high-quality conversations)

2. **Reward Model**: Trains a reward model on the UltraFeedback dataset (~60K samples) to score response quality

3. **DPO (Direct Preference Optimization)**: Aligns the model with human preferences using chosen/rejected pairs from UltraFeedback

4. **GRPO (Group Relative Policy Optimization)**: Advanced RL training on mathematical reasoning using DeepMath-103K dataset

## Features

- **Four Training Methods**: Complete coverage of modern LLM training techniques
- **Memory Efficient**: Optimized for 8GB VRAM using Liger Kernel (60% memory reduction)
- **Single Command Execution**: Run the entire pipeline with `python run_all.py`
- **TensorBoard Logging**: Real-time training metrics and visualization
- **Production-Ready Configs**: Gradient checkpointing, bf16 precision, optimized data loading

## Requirements

### Hardware

- NVIDIA GPU with **6GB+ VRAM** (RTX 3050, RTX 2060, etc.)
- 8GB+ system RAM recommended
- 20GB+ disk space for models and checkpoints

### Software

- Python 3.8+
- CUDA 11.8+ or CUDA 12.x
- Linux or WSL2 recommended

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| trl | >=0.26.0 | Transformer Reinforcement Learning trainers |
| transformers | >=4.47.0 | Model loading and tokenization |
| datasets | latest | Hugging Face datasets |
| accelerate | >=1.1.0 | Multi-GPU and mixed precision |
| peft | >=0.13.0 | Parameter-efficient fine-tuning |
| torch | >=2.2.0 | PyTorch framework |
| tensorboard | latest | Training visualization |
| liger-kernel | latest | Memory-efficient CUDA kernels |

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that can significantly speed up dependency installation.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd llmrl

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd llmrl

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Complete Pipeline

Execute all four training stages sequentially:

```bash
python run_all.py
```

### Run Individual Scripts

Train specific components:

```bash
# Supervised Fine-Tuning
python train_sft.py

# Reward Model Training
python train_reward.py

# Direct Preference Optimization
python train_dpo.py

# Group Relative Policy Optimization
python train_grpo.py
```

## Training Scripts

### `train_sft.py` - Supervised Fine-Tuning

| Parameter | Value |
|-----------|-------|
| **Trainer** | `SFTTrainer` |
| **Model** | `Qwen/Qwen2.5-0.5B` |
| **Dataset** | `trl-lib/Capybara` (~16K samples) |
| **Batch Size** | 8 per device |
| **Output** | `Qwen2.5-0.5B-SFT/` |

Trains a base model to follow instructions using high-quality conversation data.

### `train_reward.py` - Reward Model

| Parameter | Value |
|-----------|-------|
| **Trainer** | `RewardTrainer` |
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Dataset** | `trl-lib/ultrafeedback_binarized` (~60K samples) |
| **Batch Size** | 8 per device |
| **Output** | `Qwen2.5-0.5B-Reward/` |

Trains a reward model to score response quality based on human preferences.

### `train_dpo.py` - Direct Preference Optimization

| Parameter | Value |
|-----------|-------|
| **Trainer** | `DPOTrainer` |
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Dataset** | `trl-lib/ultrafeedback_binarized` (~60K samples) |
| **Batch Size** | 2 per device (dual model in memory) |
| **Output** | `Qwen2.5-0.5B-DPO/` |

Aligns the model with human preferences without requiring a separate reward model during training.

### `train_grpo.py` - Group Relative Policy Optimization

| Parameter | Value |
|-----------|-------|
| **Trainer** | `GRPOTrainer` |
| **Model** | `Qwen/Qwen2-0.5B-Instruct` |
| **Dataset** | `trl-lib/DeepMath-103K` (~103K samples) |
| **Reward Function** | `accuracy_reward` |
| **Batch Size** | 2 per device (generation overhead) |
| **Output** | `Qwen2-0.5B-GRPO/` |

Advanced RL training with verifiable rewards for mathematical reasoning tasks.

## Models & Datasets

### Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 500M | Base model for SFT |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 500M | Instruction-tuned for Reward/DPO |
| [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 500M | Used for GRPO training |

### Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| [trl-lib/Capybara](https://huggingface.co/datasets/trl-lib/Capybara) | ~16K | High-quality multi-turn conversations |
| [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | ~60K | Preference data with chosen/rejected pairs |
| [trl-lib/DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K) | ~103K | Mathematical reasoning problems |

## Memory Optimizations

This project uses several techniques to minimize GPU memory usage:

### Liger Kernel

```python
use_liger_kernel=True  # 60% memory reduction
```

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides memory-efficient CUDA implementations of common operations.

### Gradient Checkpointing

```python
gradient_checkpointing=True
```

Trades compute for memory by recomputing activations during backward pass.

### BF16 Precision

```python
bf16=True
```

Uses bfloat16 mixed precision for reduced memory and faster training.

### Batch Size Tuning

- **SFT/Reward**: Batch size 8 with gradient accumulation 4
- **DPO**: Batch size 2 (requires two models in memory)
- **GRPO**: Batch size 2 (generation overhead)

Effective batch size = `per_device_batch_size * gradient_accumulation_steps * num_gpus`

## Configuration

### Modifying Training Parameters

Edit the `*Config` objects in each training script:

```python
args=SFTConfig(
    output_dir="custom-output",
    per_device_train_batch_size=4,  # Reduce if OOM
    gradient_accumulation_steps=8,   # Increase to maintain effective batch size
    num_train_epochs=3,              # Add for multiple epochs
    learning_rate=2e-5,              # Customize learning rate
    # ... other parameters
)
```

### Using Different Models

Replace the model identifier in the training scripts:

```python
model="meta-llama/Llama-3.2-1B"  # Or any compatible model
```

### Using Custom Datasets

Load your own dataset:

```python
from datasets import load_dataset

# From Hugging Face Hub
dataset = load_dataset("your-username/your-dataset", split="train")

# From local files
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
```

## Monitoring Training

### TensorBoard

All training scripts log to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir .

# View in browser
# http://localhost:6006
```

### Training Logs

Each script creates a `training.log` file in its output directory:

```
Qwen2.5-0.5B-SFT/training.log
Qwen2.5-0.5B-Reward/training.log
Qwen2.5-0.5B-DPO/training.log
Qwen2-0.5B-GRPO/training.log
```

## Output Structure

After training, each script creates:

```
<output_dir>/
├── training.log          # Console output log
├── logs/                  # TensorBoard logs
├── config.json           # Model config
├── model.safetensors     # Model weights
├── tokenizer.json        # Tokenizer
└── ...                   # Other model files
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` proportionally
3. Ensure `gradient_checkpointing=True`
4. Ensure `use_liger_kernel=True`

### Slow Training

1. Ensure CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check GPU utilization: `nvidia-smi`
3. Verify `dataloader_num_workers` matches your CPU cores

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## References

- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [TRL GitHub Repository](https://github.com/huggingface/trl)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [GRPO Paper (DeepSeek-R1)](https://huggingface.co/papers/2402.03300)
- [DPO Paper](https://huggingface.co/papers/2305.18290)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the TRL library and transformers ecosystem
- [Qwen Team](https://huggingface.co/Qwen) for the Qwen model family
- [LinkedIn](https://github.com/linkedin/Liger-Kernel) for the Liger Kernel

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
