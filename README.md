# LLM RL - Language Model Reinforcement Learning Pipeline

A complete, memory-efficient pipeline for training language models using reinforcement learning techniques. Built on [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) from Hugging Face.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![TRL](https://img.shields.io/badge/TRL-0.26.0+-green.svg)

## Overview

A complete LLM training pipeline covering SFT, Reward Modeling, DPO, and GRPO. All scripts are optimized for **6GB VRAM GPUs** using Liger Kernel and gradient checkpointing.

## Project Structure

```
llmrl/
├── algorithms/              # Training algorithm implementations
│   ├── __init__.py          # Module exports
│   ├── base.py              # TrainingConfig dataclass + shared utilities
│   ├── sft.py               # Supervised Fine-Tuning
│   ├── reward.py            # Reward Model Training
│   ├── dpo.py               # Direct Preference Optimization
│   └── grpo.py              # Group Relative Policy Optimization
├── utils/                   # Logging and visualization utilities
│   ├── __init__.py          # Module exports
│   ├── logging.py           # Rich console output and progress tracking
│   └── run_id.py            # Run ID generation and directory management
├── cache_config.py          # HuggingFace cache directory configuration
├── pipeline.py              # Runs all algorithms for a given config
├── qwen2.5_0.5b.py          # Qwen 2.5 0.5B production configs
├── smollm2_135m.py          # SmolLM2-135M test configs (fast validation)
├── run_all.py               # Main entry point
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template
```

## Training Pipeline

```
+---------------------------------------------------------------------+
|                     LLM RL Training Pipeline                        |
+---------------------------------------------------------------------+
|                                                                     |
|  +----------+    +-----------+    +----------+    +-----------+     |
|  |   SFT    |--->|  Reward   |--->|   DPO    |--->|   GRPO    |     |
|  | (16K)    |    |  (62K)    |    |  (62K)   |    |  (103K)   |     |
|  +----------+    +-----------+    +----------+    +-----------+     |
|                                                                     |
|  Instruction     Response         Preference      Reinforcement     |
|  Following       Quality          Alignment       Learning          |
|                                                                     |
+---------------------------------------------------------------------+
```

### Training Stages

1. **SFT (Supervised Fine-Tuning)**: Teaches the model to follow instructions
2. **Reward Model**: Trains a model to score response quality
3. **DPO (Direct Preference Optimization)**: Aligns the model with human preferences
4. **GRPO (Group Relative Policy Optimization)**: Advanced RL for mathematical reasoning

## Features

- **Memory Efficient**: Optimized for 6GB+ VRAM using Liger Kernel (60% memory reduction)
- **Modular Architecture**: Algorithms and configs cleanly separated
- **Test Pipeline**: Fast validation with SmolLM2-135M
- **Experiment Tracking**: TensorBoard, Weights & Biases, and Neptune integration
- **Rich Console Output**: Progress bars, tables, and color-coded messages via [Rich](https://rich.readthedocs.io/)
- **GPU Memory Monitoring**: Live GPU memory display during training
- **Unique Run IDs**: Timestamped IDs (YYYYMMDD_HHMMSS_xxxx) prevent overwrites
- **Comprehensive Logging**: Metrics saved to JSONL, events log, and console output

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

trl, transformers, datasets, accelerate, peft, torch, tensorboard, liger-kernel, rich, psutil, wandb, neptune, python-dotenv

## Installation

### Using uv (Recommended)

```bash
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

Test the entire pipeline with SmolLM2-135M (~1-2 minutes):

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
# Test pipeline (SmolLM2-135M)
python smollm2_135m.py

# Production pipeline (Qwen 2.5 0.5B)
python qwen2.5_0.5b.py
```

## Models & Datasets

### Models

| Model | Parameters | Disk Size | Use Case |
|-------|------------|-----------|----------|
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 490M | ~988 MB | Base model for SFT |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 490M | ~988 MB | Instruction-tuned for Reward/DPO/GRPO |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | ~724 MB | Fast testing (all algorithms) |

### Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| [trl-lib/Capybara](https://huggingface.co/datasets/trl-lib/Capybara) | ~16K | High-quality multi-turn conversations |
| [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | ~62K | Preference data with chosen/rejected pairs |
| [trl-lib/DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K) | ~103K | Mathematical reasoning problems |

## Memory Optimizations

This project uses several techniques to minimize GPU memory usage:

### Liger Kernel

```python
use_liger_kernel=True  # 60% memory reduction
```

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides memory-efficient Triton kernels for LLM training operations.

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

### GRPO Configuration

GRPO supports custom reward functions and generation parameters:

```python
from algorithms.grpo import GRPOExtraConfig

grpo_extra = GRPOExtraConfig(
    reward_func=None,           # Uses accuracy_reward by default
    num_generations=4,          # Completions per prompt
    max_completion_length=256,  # Max tokens per completion
)

run_pipeline(configs, grpo_extra=grpo_extra)
```

### Environment Variables

Copy `.env.example` to `.env` and configure your API keys for experiment tracking:

```bash
# Weights & Biases
WANDB_API_KEY=your-key
WANDB_PROJECT=llmrl

# Neptune.ai
NEPTUNE_API_TOKEN=your-token
NEPTUNE_PROJECT=your-workspace/llmrl
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

Each run creates comprehensive logs in the `runs/` directory:

```
runs/
└── 20250111_143052_ab12/     # Unique run ID
    ├── run_info.json         # Run metadata (algorithms, timestamp)
    ├── Qwen2.5-0.5B-SFT/     # SFT output
    ├── Qwen2.5-0.5B-Reward/  # Reward output
    ├── Qwen2.5-0.5B-DPO/     # DPO output
    └── Qwen2.5-0.5B-GRPO/    # GRPO output
```

## Output Structure

After training, each algorithm creates:

```
<output_dir>/
├── training.log           # Standard training log
├── console_output.log     # Full Rich console output (tables, progress bars)
├── training_metrics.jsonl # Step-by-step metrics in JSON Lines format
├── training_events.log    # Human-readable event log
├── logs/                  # TensorBoard logs
├── config.json            # Model config
├── model.safetensors      # Model weights
├── tokenizer.json         # Tokenizer
└── ...                    # Other model files
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

- [Hugging Face](https://huggingface.co/) for the TRL library, transformers ecosystem, and [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) models
- [Qwen Team](https://huggingface.co/Qwen) for the Qwen model family
- [LinkedIn](https://github.com/linkedin/Liger-Kernel) for the Liger Kernel

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
