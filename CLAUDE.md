# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIpparel is a multimodal foundation model for digital garments (CVPR 2025 Highlight). It processes images and text to generate sewing patterns represented as JSON specifications.

## Key Architecture Components

- **Model**: Extends LLaVA (Large Language and Vision Assistant) v1.5-7B with custom garment tokenization
- **Data Processing**: Handles GarmentCodeData (GCD) dataset with multimodal annotations
- **Tokenization**: Custom garment tokenizer for sewing pattern representation
- **Training**: Uses DeepSpeed with distributed training support

## Common Commands

### Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.10
uv venv --python 3.10

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install -r requirements.txt

# Add project directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/AIpparel-Code
```

### Training
```bash
# Multi-GPU training with 4 GPUs
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/run.py \
  experiment.project_name=AIpparel \
  experiment.run_name=train \
  evaluate=False \
  --config-name aipparel --config-path ../configs
```

### Inference
```bash
# Single GPU inference
python scripts/inference.py --config-name aipparel_inference --config-path ../configs
```

### Evaluation
```bash
# Full evaluation on test set
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/run.py \
  experiment.project_name=AIpparel \
  experiment.run_name=eval_multimodal \
  pre_trained=/path/to/aipparel_pretrained.pth \
  evaluate=True \
  --config-name aipparel --config-path ../configs
```

### Data Preprocessing
```bash
# Process sewing patterns (update gcd_path in script first)
python shift_specs.py
```

## Configuration System

Uses Hydra configuration framework with hierarchical configs:
- Main config: `configs/aipparel.yaml`
- Data config: `configs/data_wrapper/`
- Model config: `configs/model/`
- Training config: `configs/trainer/`
- Experiment config: `configs/experiment/`

Key configuration parameters:
- `pre_trained`: Path to pre-trained model weights
- `model_max_length`: Maximum sequence length (default: 2100)
- `precision`: Training precision (bf16/fp16)
- `evaluate`: Toggle between training and evaluation mode

## Data Structure

- Sewing patterns stored as JSON specifications with panels, vertices, edges, and stitches
- Images processed through CLIP vision encoder
- Text instructions for garment editing tasks
- Panel classification system defined in `assets/data_configs/panel_classes_garmentcodedata.json`

## Development Notes

- No unit tests currently exist in the codebase
- Logging handled via Weights & Biases (configure in `configs/experiment/wandb_info/wandb.yaml`)
- Windows DLLs for Cairo graphics included for pattern visualization
- Uses custom conversation templates from LLaVA for multimodal dialogue