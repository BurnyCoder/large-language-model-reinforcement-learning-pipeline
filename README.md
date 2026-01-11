# LLM RL - Language Model Reinforcement Learning Pipeline

A complete, memory-efficient pipeline for training language models using reinforcement learning techniques. Built on [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) from Hugging Face.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![TRL](https://img.shields.io/badge/TRL-0.26.0+-green.svg)

## Overview

This project demonstrates a complete LLM training pipeline with four key training methods:

- **Supervised Fine-Tuning (SFT)** - Train models on high-quality instruction data
- **Reward Modeling** - Train models to evaluate response quality
- **Direct Preference Optimization (DPO)** - Align models with human preferences
- **Group Relative Policy Optimization (GRPO)** - Memory-efficient reinforcement learning

All scripts are optimized for **6GB VRAM GPUs** using memory-efficient techniques like Liger Kernel and gradient checkpointing.

## Project Structure

```
llmrl/
├── algorithms/              # Training algorithm implementations
│   ├── base.py              # TrainingConfig dataclass + shared utilities
│   ├── sft.py               # Supervised Fine-Tuning
│   ├── reward.py            # Reward Model Training
│   ├── dpo.py               # Direct Preference Optimization
│   └── grpo.py              # Group Relative Policy Optimization
├── utils/                   # Logging and visualization utilities
│   └── logging.py           # Rich console output and progress tracking
├── pipeline.py              # Runs all algorithms for a given config
├── qwen2.5_0.5.py           # Qwen 2.5 0.5B production configs
├── tiny_gpt2.py             # tiny-gpt2 test configs (fast validation)
├── run_all.py               # Main entry point
└── requirements.txt
```

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
- **Memory Efficient**: Optimized for 6GB+ VRAM using Liger Kernel (60% memory reduction)
- **Modular Architecture**: Algorithms and configs cleanly separated
- **Test Pipeline**: Fast validation with tiny-gpt2 (~1-2 minutes)
- **TensorBoard Logging**: Real-time training metrics and visualization
- **Production-Ready Configs**: Gradient checkpointing, bf16 precision, optimized data loading

## Requirements

### Hardware

- NVIDIA GPU with **6GB+ VRAM** (RTX 3050, RTX 2060, etc.)
- 8GB+ system RAM recommended
- 20GB+ disk space for models and checkpoints

### Software

- Python 3.10+
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
| rich | >=13.0.0 | Beautiful console output and progress bars |
| psutil | >=5.9.0 | System resource monitoring |

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

### Run Test Pipeline (Fast Validation)

Test the entire pipeline with tiny-gpt2 (~1-2 minutes):

```bash
python run_all.py --test
```

### Run Production Pipeline

Train with Qwen 2.5 0.5B models:

```bash
python run_all.py --prod
```

### Run Both Pipelines

```bash
python run_all.py
```

### Run Individual Model Pipelines

```bash
# Test pipeline (tiny-gpt2)
python tiny_gpt2.py

# Production pipeline (Qwen 2.5 0.5B)
python qwen2.5_0.5.py
```

## Model Configurations

### Production: `qwen2.5_0.5.py`

| Algorithm | Model | Dataset | Batch Size |
|-----------|-------|---------|------------|
| SFT | `Qwen/Qwen2.5-0.5B` | `trl-lib/Capybara` | 8 |
| Reward | `Qwen/Qwen2.5-0.5B-Instruct` | `trl-lib/ultrafeedback_binarized` | 8 |
| DPO | `Qwen/Qwen2.5-0.5B-Instruct` | `trl-lib/ultrafeedback_binarized` | 2 |
| GRPO | `Qwen/Qwen2.5-0.5B-Instruct` | `trl-lib/DeepMath-103K` | 2 |

### Test: `tiny_gpt2.py`

| Setting | Value |
|---------|-------|
| Model | `sshleifer/tiny-gpt2` (~100K params, ~4.7 MB) |
| Max Steps | 10 |
| Max Samples | 10 |
| Save Steps | 5 |
| Batch Size | 1 |

## Models & Datasets

### Models

| Model | Parameters | Disk Size | Use Case |
|-------|------------|-----------|----------|
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 490M | ~988 MB | Base model for SFT |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 490M | ~988 MB | Instruction-tuned for Reward/DPO/GRPO |
| [sshleifer/tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2) | ~100K | ~4.7 MB | Fast testing and CI validation |

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

### Adding a New Model Configuration

Create a new file (e.g., `llama_1b.py`):

```python
from algorithms import TrainingConfig
from pipeline import run_pipeline

MODEL = "meta-llama/Llama-3.2-1B"

configs = {
    "sft": TrainingConfig(
        model_name=MODEL,
        output_dir="Llama-1B-SFT",
        dataset_name="trl-lib/Capybara",
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=3,
    ),
    # ... add other algorithms
}

if __name__ == "__main__":
    run_pipeline(configs)
```

### Using Custom Datasets

```python
TrainingConfig(
    model_name="your-model",
    output_dir="output",
    dataset_name="your-username/your-dataset",
    # ...
)
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
Qwen2.5-0.5B-GRPO/training.log
```

## Output Structure

After training, each algorithm creates:

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
- [GRPO Paper (DeepSeekMath)](https://huggingface.co/papers/2402.03300)
- [DPO Paper](https://huggingface.co/papers/2305.18290)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the TRL library and transformers ecosystem
- [Qwen Team](https://huggingface.co/Qwen) for the Qwen model family
- [LinkedIn](https://github.com/linkedin/Liger-Kernel) for the Liger Kernel

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
