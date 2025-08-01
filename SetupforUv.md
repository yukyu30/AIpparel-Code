# AIpparel Environment Setup with uv

This guide provides step-by-step instructions for setting up the AIpparel environment using `uv` package manager.

## Prerequisites

- Python 3.10 installed on your system
- NVIDIA GPU with CUDA support (for training/inference)
- Git for cloning the repository

## Installation Steps

### 1. Install uv

#### Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/georgeNakayama/AIpparel-Code.git
cd AIpparel-Code
```

### 3. Create Virtual Environment

```bash
# Create a virtual environment with Python 3.10
uv venv --python 3.10
```

### 4. Activate Virtual Environment

#### Windows
```bash
.venv\Scripts\activate
```

#### Linux/macOS
```bash
source .venv/bin/activate
```

### 5. Install PyTorch with CUDA 12.1

```bash
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 6. Install Dependencies

```bash
# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

**Note for Windows users**: DeepSpeed and flash_attn may fail to install due to build issues. You can install other dependencies by excluding these packages:

```bash
# Windows alternative (excluding problematic packages)
uv pip install CairoSVG==2.7.1 einops==0.8.1 huggingface_hub==0.23.4 hydra-core==1.3.2 matplotlib==3.10.1 numpy==1.25.0 omegaconf==2.3.0 opencv_python==4.10.0.84 packaging==25.0 peft==0.12.0 Pillow==11.2.1 Requests==2.32.3 scipy==1.15.2 svgpathtools==1.6.1 tqdm==4.66.4 transformers==4.31.0 wandb==0.19.10 sentencepiece==0.1.99
```

### 7. Set Environment Variables

#### Windows
```bash
set PYTHONPATH=%PYTHONPATH%;C:\path\to\AIpparel-Code
```

#### Linux/macOS
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/AIpparel-Code
```

## Running Inference

### 1. Download Pre-trained Weights

Download the pre-trained model weights from [Hugging Face](https://huggingface.co/georgeNakayama/AIpparel):
- `aipparel_pretrained.pth` - Main model weights

Place the downloaded file in the project root directory.

### 2. Update Configuration

Edit `configs/aipparel_inference.yaml` to update the model path:
```yaml
pre_trained: aipparel_pretrained.pth  # or full path to the weights
```

### 3. Run Inference

```bash
# Windows
python scripts/inference.py --config-name aipparel_inference --config-path ../configs

# Linux/macOS
python scripts/inference.py --config-name aipparel_inference --config-path ../configs
```

## Troubleshooting

### CUDA Out of Memory Error

If you encounter GPU memory issues:

1. **Check GPU memory usage**:
   ```bash
   nvidia-smi
   ```

2. **Clear GPU memory**:
   - Close unnecessary applications using GPU
   - Restart Python processes if needed

3. **Reduce memory usage**:
   - The model already uses bf16 precision and batch size 1
   - Consider using a smaller image resolution if possible

### Module Import Errors

Ensure PYTHONPATH is set correctly:
```bash
# Windows
echo %PYTHONPATH%

# Linux/macOS
echo $PYTHONPATH
```

### Windows-Specific Issues

- **DeepSpeed**: Required for training but not for inference. For training on Windows, use WSL2 or Linux.
- **flash_attn**: Optional performance optimization. Not required for basic inference.

## Training Setup

For training, you'll need:
1. Complete dataset setup (GarmentCodeData)
2. DeepSpeed installation (Linux/WSL2 recommended)
3. Multiple GPUs for distributed training

Refer to the main README.md for detailed training instructions.

## Additional Resources

- [Project Page](https://georgenakayama.github.io/AIpparel/)
- [Paper](https://arxiv.org/abs/2412.03937)
- [Hugging Face Model](https://huggingface.co/georgeNakayama/AIpparel)