# 💿 Installation Guide

Complete installation guide for **Fabgaurd-AI** defect detection system.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Step-by-Step Installation](#step-by-step-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Docker Installation](#docker-installation-optional)
- [Cloud Setup](#cloud-setup)

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 10/11, Ubuntu 18.04+, macOS 10.15+ |
| **Python** | 3.8 or higher |
| **RAM** | 8 GB |
| **Storage** | 5 GB free space |
| **CPU** | 4 cores, 2.0 GHz |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 11, Ubuntu 20.04+, macOS 12+ |
| **Python** | 3.9 - 3.11 |
| **RAM** | 16 GB or more |
| **Storage** | 20 GB SSD |
| **GPU** | NVIDIA GPU with 6GB+ VRAM |
| **CUDA** | 11.8 or higher |

### GPU Support

**NVIDIA GPUs** (Recommended for training):
- CUDA 11.0 or higher
- cuDNN 8.0 or higher
- Compute Capability 3.5 or higher

**Apple Silicon** (M1/M2/M3):
- Metal Performance Shaders (MPS) supported
- Native ARM64 builds available

**AMD GPUs**:
- ROCm support (experimental)

---

## Installation Methods

Choose the installation method that best suits your needs:

1. **Standard Installation** (Recommended): Virtual environment with pip
2. **Conda Installation**: Using Anaconda/Miniconda
3. **Docker Installation**: Containerized deployment
4. **Development Installation**: Editable install for contributors

---

## Step-by-Step Installation

### Method 1: Standard Installation (pip + venv)

#### Step 1: Install Python

**Windows**:
```bash
# Download from python.org or use winget
winget install Python.Python.3.11
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**macOS**:
```bash
# Using Homebrew
brew install python@3.11
```

#### Step 2: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI

# Or using SSH
git clone git@github.com:adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
```

#### Step 3: Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

#### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

#### Step 5: Install PyTorch

**For CUDA (NVIDIA GPU)**:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU Only**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For Apple Silicon (M1/M2/M3)**:
```bash
# MPS acceleration is built-in for Apple Silicon
pip install torch torchvision torchaudio
```

#### Step 6: Install Project Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- scikit-learn
- numpy
- Pillow
- onnx
- onnxruntime
- matplotlib
- tqdm
- pandas

#### Step 7: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

Expected output:
```
PyTorch version: 2.1.2+cu118
CUDA available: True
TorchVision version: 0.16.2+cu118
```

---

### Method 2: Conda Installation

#### Step 1: Install Anaconda/Miniconda

Download from: https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html

#### Step 2: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n fabgaurd python=3.10
conda activate fabgaurd
```

#### Step 3: Clone Repository

```bash
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
```

#### Step 4: Install PyTorch via Conda

**For CUDA**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For CPU Only**:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Method 3: Docker Installation (Optional)

#### Prerequisites
- Docker Desktop 20.10+
- NVIDIA Docker (for GPU support)

#### Step 1: Pull or Build Image

**Option A: Pull from Docker Hub** (when available):
```bash
docker pull fabgaurd/fabgaurd-ai:latest
```

**Option B: Build from Dockerfile**:
```bash
cd Fabgaurd-AI
docker build -t fabgaurd-ai:latest .
```

#### Step 2: Run Container

**CPU Version**:
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fabgaurd-ai:latest
```

**GPU Version** (Requires NVIDIA Docker):
```bash
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fabgaurd-ai:latest
```

---

## Verification

### Test Installation

Create a test script `test_installation.py`:

```python
import torch
import torchvision
from torchvision import models, transforms
import sys

def test_installation():
    print("=" * 60)
    print("Fabgaurd-AI Installation Test")
    print("=" * 60)
    
    # Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ TorchVision version: {torchvision.__version__}")
    
    # Device check
    if torch.cuda.is_available():
        print(f"✓ CUDA available: Yes")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✓ MPS (Apple Silicon) available: Yes")
    else:
        print(f"✓ Using CPU")
    
    # Test model loading
    try:
        model = models.efficientnet_b0(weights=None)
        print(f"✓ EfficientNet-B0 loaded successfully")
    except Exception as e:
        print(f"✗ Error loading EfficientNet-B0: {e}")
        return False
    
    try:
        model = models.mobilenet_v3_small(weights=None)
        print(f"✓ MobileNetV3-Small loaded successfully")
    except Exception as e:
        print(f"✗ Error loading MobileNetV3-Small: {e}")
        return False
    
    # Test transforms
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        print(f"✓ Transforms working correctly")
    except Exception as e:
        print(f"✗ Error with transforms: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Installation Test: PASSED ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
```

Run the test:
```bash
python test_installation.py
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size in training scripts
2. Use gradient accumulation
3. Enable mixed precision training

```python
# In train_teacher.py or train_student.py
BATCH_SIZE = 16  # Reduce from 32
```

#### Issue 2: Import Error for torch

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solutions**:
1. Ensure virtual environment is activated
2. Reinstall PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 3: Permission Denied (Linux/macOS)

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
```bash
# Fix permissions
sudo chown -R $USER:$USER ~/Fabgaurd-AI
chmod -R 755 ~/Fabgaurd-AI
```

#### Issue 4: SSL Certificate Error

**Symptoms**:
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions**:
```bash
# Upgrade certifi
pip install --upgrade certifi

# Or temporarily disable SSL verification (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### Issue 5: Model Download Issues

**Symptoms**:
Models fail to download during first run

**Solutions**:
1. Check internet connection
2. Set proxy if behind firewall:
```bash
export http_proxy="http://proxy.example.com:8080"
export https_proxy="http://proxy.example.com:8080"
```

3. Manually download models:
```python
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Small_Weights

# Pre-download weights
EfficientNet_B0_Weights.DEFAULT
MobileNet_V3_Small_Weights.DEFAULT
```

#### Issue 6: Windows Long Path Error

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions**:
Enable long path support in Windows:
1. Run as Administrator:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. Or use shorter directory path

---

## Development Installation

For developers who want to contribute:

```bash
# Clone repository
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Cloud Setup

### AWS Setup

```bash
# Launch EC2 instance (Deep Learning AMI)
# Instance type: g4dn.xlarge or higher

# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Clone and setup
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
pip install -r requirements.txt
```

### Google Cloud Platform

```bash
# Create Compute Engine instance with GPU
gcloud compute instances create fabgaurd-ai \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# SSH and setup
gcloud compute ssh fabgaurd-ai
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
pip install -r requirements.txt
```

### Azure Setup

```bash
# Create Azure ML workspace and compute instance
# Follow Azure ML documentation

# Clone repository in Azure ML
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
pip install -r requirements.txt
```

---

## Post-Installation Setup

### Dataset Preparation

After installation, prepare your dataset:

```bash
# Create data directories (if not exist)
mkdir -p data/train data/val data/test

# Organize your images into class folders
# See docs/DATASET.md for detailed structure
```

### Download Pre-trained Models (Optional)

```bash
# Download teacher model
wget https://example.com/models/teacher_b0_refined.pth -O models/teacher_b0_refined.pth

# Download student model
wget https://example.com/models/student_mobilenet_small.pth -O models/student_mobilenet_small.pth
```

---

## Updating Fabgaurd-AI

### Update from Git

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Update PyTorch

```bash
# Check for updates
pip list --outdated | grep torch

# Update (example for CUDA 11.8)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Uninstallation

### Remove Virtual Environment

```bash
# Deactivate environment
deactivate

# Remove environment directory
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Remove Conda Environment

```bash
conda deactivate
conda env remove -n fabgaurd
```

### Remove Project Files

```bash
cd ..
rm -rf Fabgaurd-AI
```

---

## Next Steps

After successful installation:

1. 📖 Read [USAGE.md](USAGE.md) for usage instructions
2. 📊 Review [DATASET.md](DATASET.md) for dataset preparation
3. 🏗️ Check [ARCHITECTURE.md](ARCHITECTURE.md) for system details
4. 🚀 Start training your first model!

---

## Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search [GitHub Issues](https://github.com/adarshshkla/Fabgaurd-AI/issues)
3. Create a new issue with:
   - OS and Python version
   - Error message and stack trace
   - Steps to reproduce

---

*Last Updated: February 2026*
