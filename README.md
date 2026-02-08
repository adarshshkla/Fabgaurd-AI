<div align="center">

# 🛡️ Fabgaurd-AI

### *"Guarding Silicon, One Die at a Time."*

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/Computer_Vision-AI-green.svg" alt="CV">
<img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">

**An intelligent AI-powered defect detection system for semiconductor and PCB manufacturing quality control**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [System Architecture](#-architecture)
- [Defect Categories](#-defect-categories)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Fabgaurd-AI** is a state-of-the-art deep learning solution designed to revolutionize quality control in semiconductor and PCB (Printed Circuit Board) manufacturing. By leveraging advanced computer vision techniques and knowledge distillation, Fabgaurd-AI provides:

- **High-accuracy defect detection** across 12 different defect categories
- **Real-time inference** optimized for edge deployment
- **Knowledge distillation pipeline** for model compression without significant accuracy loss
- **Production-ready system** with quantized models for industrial deployment

The system addresses critical challenges in semiconductor manufacturing where even microscopic defects can lead to catastrophic failures, ensuring only the highest quality products reach the market.

---

## ✨ Features

### 🧠 **Dual-Model Architecture**
- **Teacher Model**: EfficientNet-B0 for maximum accuracy and robust feature learning
- **Student Model**: MobileNetV3-Small for edge deployment with minimal latency
- **Knowledge Distillation**: Transfer learning from teacher to student for optimal performance

### 🎯 **Comprehensive Defect Detection**
Detects 12 types of manufacturing defects:
- Bridge, Crack, Delamination, Gap
- Open Circuit, Short Circuit, VIAS defects
- Particle contamination, Polishing issues
- Void formation, Random anomalies
- Good/Pass classification

### ⚡ **Optimized for Production**
- **INT8 Quantization**: 4x smaller model size with minimal accuracy loss
- **ONNX Export**: Cross-platform deployment capability
- **Balanced Dataset Handling**: Weighted sampling for imbalanced classes
- **GPU/CPU/MPS Support**: Automatic device detection and optimization

### 🛠️ **Robust Training Pipeline**
- Advanced data augmentation strategies
- Label smoothing for better generalization
- Cosine annealing and adaptive learning rate scheduling
- Early stopping and model checkpointing
- Comprehensive validation metrics

---

## 🏗️ Architecture

Fabgaurd-AI employs a sophisticated two-stage training architecture:

```mermaid
graph LR
    A[Raw Dataset] --> B[Teacher Model Training]
    B --> C[EfficientNet-B0 Teacher]
    C --> D[Knowledge Distillation]
    A --> D
    D --> E[MobileNetV3-Small Student]
    E --> F[Model Quantization]
    F --> G[INT8 ONNX Model]
    G --> H[Edge Deployment]
```

### **Stage 1: Teacher Training**
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Purpose**: Learn rich feature representations
- **Training**: 30 epochs with AdamW optimizer
- **Augmentation**: Moderate augmentation to preserve context

### **Stage 2: Knowledge Distillation**
- **Model**: MobileNetV3-Small (tiny architecture)
- **Technique**: Soft target distillation with temperature scaling
- **Loss Function**: Combined soft loss (KL divergence) + hard loss (cross-entropy)
- **Benefit**: 85-90% of teacher accuracy with 10x smaller model

### **Stage 3: Production Optimization**
- **Quantization**: INT8 post-training quantization
- **Export**: ONNX format for universal deployment
- **Optimization**: Optimized for inference speed and memory efficiency

---

## 🔍 Defect Categories

The system is trained to identify the following defect types commonly found in semiconductor and PCB manufacturing:

| Category | Description | Severity |
|----------|-------------|----------|
| **Bridge** | Unwanted electrical connections between conductors | Critical |
| **Crack** | Physical fractures in substrate or traces | Critical |
| **Delamination** | Layer separation in multi-layer boards | Critical |
| **Gap** | Incomplete connections or broken traces | Critical |
| **Open** | Discontinuity in electrical path | Critical |
| **Short** | Unintended short circuits | Critical |
| **VIAS** | Defects in through-hole connections | High |
| **Void** | Air pockets or missing material | High |
| **Particle** | Foreign material contamination | Medium |
| **Polishing** | Surface finish irregularities | Medium |
| **Random** | Unclassified anomalies | Variable |
| **Good** | No defects detected (pass) | N/A |

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (optional, for GPU acceleration)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for models and dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/adarshshkla/Fabgaurd-AI.git
cd Fabgaurd-AI
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎮 Quick Start

### Dataset Preparation

Organize your dataset in the following structure:

```
Fabgaurd-AI/
├── train/
│   ├── Bridge/
│   ├── Crack/
│   ├── Delamination/
│   ├── Gap/
│   ├── Good/
│   ├── Open/
│   ├── Particle/
│   ├── Polishing/
│   ├── Random/
│   ├── Short/
│   ├── VIAS/
│   └── Void/
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

### Training the Teacher Model

```bash
cd src
python train_teacher.py
```

**Configuration** (edit in `train_teacher.py`):
- `BATCH_SIZE`: 32 (adjust based on GPU memory)
- `EPOCHS`: 30
- `LEARNING_RATE`: 1e-3

**Expected Output**:
- Training progress with F1 scores per epoch
- Best model saved to `models/teacher_b0_refined.pth`
- Detailed classification report

### Training the Student Model (Knowledge Distillation)

```bash
python train_student.py
```

**Configuration** (edit in `train_student.py`):
- `TEMPERATURE`: 4.0 (softening factor for distillation)
- `ALPHA`: 0.7 (balance between soft and hard loss)
- `EPOCHS`: 30

**Expected Output**:
- Student model training progress
- Best model saved to `models/student_mobilenet_small.pth`
- Comparative performance metrics

---

## 📁 Project Structure

```
Fabgaurd-AI/
├── 📄 README.md                    # Main documentation (you are here!)
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 CHANGELOG.md                 # Version history
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 src/                         # Source code
│   ├── train_teacher.py            # Teacher model training script
│   └── train_student.py            # Student model distillation script
│
├── 📂 models/                      # Trained models
│   ├── teacher_b0_refined.pth      # EfficientNet-B0 teacher
│   ├── student_mobilenet_small.pth # MobileNetV3-Small student
│   ├── student_int8.onnx           # Quantized ONNX model
│   └── tiny_model_quantized.pth    # Quantized PyTorch model
│
├── 📂 data/                        # Dataset (not included in repo)
│   ├── train/                      # Training images
│   ├── val/                        # Validation images
│   └── test/                       # Test images
│
├── 📂 docs/                        # Detailed documentation
│   ├── ARCHITECTURE.md             # System architecture details
│   ├── INSTALLATION.md             # Installation guide
│   ├── USAGE.md                    # Usage instructions
│   ├── MODEL_DETAILS.md            # Model architecture documentation
│   ├── DATASET.md                  # Dataset information
│   └── API_REFERENCE.md            # API documentation
│
├── 📂 scripts/                     # Utility scripts
│   └── (conversion, evaluation scripts)
│
└── 📂 notebooks/                   # Jupyter notebooks
    └── (exploration and analysis notebooks)
```

---

## 📊 Model Performance

### Teacher Model (EfficientNet-B0)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~95% |
| **Macro F1-Score** | ~94% |
| **Model Size** | ~21 MB |
| **Inference Time** | ~45ms (GPU) |
| **Parameters** | ~5.3M |

### Student Model (MobileNetV3-Small)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~92% |
| **Macro F1-Score** | ~90% |
| **Model Size** | ~9 MB |
| **Inference Time** | ~15ms (GPU) |
| **Parameters** | ~2.5M |

### Quantized Model (INT8)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~90% |
| **Macro F1-Score** | ~88% |
| **Model Size** | ~2.3 MB |
| **Inference Time** | ~8ms (CPU) |
| **Speedup** | 4-6x over FP32 |

*Note: Performance metrics may vary based on dataset and hardware*

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[📐 ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed system architecture and design decisions
- **[💿 INSTALLATION.md](docs/INSTALLATION.md)**: Step-by-step installation guide
- **[📖 USAGE.md](docs/USAGE.md)**: Usage examples and best practices
- **[🧠 MODEL_DETAILS.md](docs/MODEL_DETAILS.md)**: In-depth model architecture documentation
- **[📊 DATASET.md](docs/DATASET.md)**: Dataset structure and preparation guide
- **[🔌 API_REFERENCE.md](docs/API_REFERENCE.md)**: API reference and integration guide

---

## 🤝 Contributing

We welcome contributions from the community! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code of conduct
- Development setup
- Pull request process  
- Coding standards
- Testing guidelines

---

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and release notes.

**Current Version**: 1.0.0

---

## 🙏 Acknowledgments

### Technologies Used

- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[EfficientNet](https://arxiv.org/abs/1905.11946)**: Teacher model architecture
- **[MobileNetV3](https://arxiv.org/abs/1905.02244)**: Student model architecture
- **[ONNX](https://onnx.ai/)**: Model interoperability
- **[Scikit-learn](https://scikit-learn.org/)**: Evaluation metrics

### Research & Inspiration

- Hinton, G., et al. (2015). *Distilling the Knowledge in a Neural Network*
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*
- Howard, A., et al. (2019). *Searching for MobileNetV3*

### Team

Developed with ❤️ by the Fabgaurd-AI team

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/adarshshkla/Fabgaurd-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adarshshkla/Fabgaurd-AI/discussions)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Fabgaurd-AI** - *Guarding Silicon, One Die at a Time.*

</div>
