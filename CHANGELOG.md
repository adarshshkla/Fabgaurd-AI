# 📜 Changelog

All notable changes to **Fabgaurd-AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-08

### 🎉 Initial Release

The inaugural release of Fabgaurd-AI - an intelligent defect detection system for semiconductor and PCB manufacturing.

### ✨ Features

#### Core Functionality
- **Two-Stage Training Pipeline**: Teacher (EfficientNet-B0) and Student (MobileNetV3-Small) models
- **Knowledge Distillation**: Efficient knowledge transfer from teacher to student with configurable temperature and alpha
- **12-Class Defect Detection**: Bridge, Crack, Delamination, Gap, Good, Open, Particle, Polishing, Random, Short, VIAS, Void
- **Automatic Device Detection**: Seamless support for CUDA, MPS (Apple Silicon), and CPU

#### Training Features
- **Weighted Random Sampling**: Handles class imbalance automatically
- **Advanced Data Augmentation**: Geometric and photometric transformations optimized for defect preservation
- **Label Smoothing**: Prevents overconfidence and improves generalization
- **Learning Rate Scheduling**: ReduceLROnPlateau and CosineAnnealing schedulers
- **Best Model Checkpointing**: Saves best model based on F1-score
- **Comprehensive Metrics**: Precision, recall, F1-score per class and macro averages

#### Model Optimization
- **Model Quantization**: INT8 post-training quantization for 4x size reduction
- **ONNX Export**: Cross-platform deployment support
- **Mixed Precision Training**: FP16 training for faster convergence (optional)
- **Gradient Clipping**: Prevents exploding gradients

#### Inference
- **Single Image Prediction**: Fast inference with confidence scores
- **Batch Processing**: Efficient batch inference for multiple images
- **Top-K Predictions**: Returns top-3 most likely classes
- **Probability Distributions**: Optional full probability output

### 📁 Project Structure

```
Fabgaurd-AI/
├── src/                          # Source code
│   ├── train_teacher.py          # Teacher model training
│   └── train_student.py          # Student model distillation
├── models/                       # Trained models
│   ├── teacher_b0_refined.pth
│   ├── student_mobilenet_small.pth
│   ├── student_int8.onnx
│   └── tiny_model_quantized.pth
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # System architecture
│   ├── INSTALLATION.md           # Installation guide
│   ├── USAGE.md                  # Usage instructions
│   ├── MODEL_DETAILS.md          # Model specifications
│   ├── DATASET.md                # Dataset guide
│   └── API_REFERENCE.md          # API documentation
├── README.md                     # Main documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

### 📊 Model Performance

#### Teacher Model (EfficientNet-B0)
- **Accuracy**: ~95%
- **F1-Score**: ~94%
- **Parameters**: 5.3M
- **Model Size**: ~20 MB
- **Inference Time (GPU)**: ~45ms

#### Student Model (MobileNetV3-Small)
- **Accuracy**: ~92%
- **F1-Score**: ~90%
- **Parameters**: 2.5M
- **Model Size**: ~10 MB
- **Inference Time (GPU)**: ~15ms
- **Speedup**: 3x faster than teacher

#### Quantized Model (INT8)
- **Accuracy**: ~90%
- **F1-Score**: ~88%
- **Model Size**: ~2.4 MB
- **Inference Time (CPU)**: ~8-12ms
- **Size Reduction**: 4x smaller

### 📚 Documentation

- Comprehensive README with badges, feature descriptions, and quick start
- Detailed architecture documentation with diagrams and mathematical formulations
- Step-by-step installation guide for Windows, Linux, and macOS
- Usage guide with code examples and best practices
- In-depth model details with architecture specifications
- Dataset preparation guide with quality assurance scripts
- API reference for integration and deployment
- Contributing guidelines for community participation

### 🛠️ Technical Specifications

#### Supported Platforms
- **Operating Systems**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **PyTorch**: 2.0+
- **CUDA**: 11.0+ (optional)

#### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Edge Deployment**: 2GB RAM, ARM CPU

### 🔧 Dependencies

Core dependencies:
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- scikit-learn >= 1.0.0
- ONNX >= 1.12.0 (optional)
- ONNXRuntime >= 1.12.0 (optional)

### 🎯 Defect Categories

1. **Bridge**: Unwanted electrical connections
2. **Crack**: Physical fractures in substrate
3. **Delamination**: Layer separation
4. **Gap**: Incomplete connections
5. **Good**: Defect-free (pass)
6. **Open**: Circuit discontinuity
7. **Particle**: Foreign contamination
8. **Polishing**: Surface finish issues
9. **Random**: Unclassified anomalies
10. **Short**: Unintended short circuits
11. **VIAS**: Via defects
12. **Void**: Air pockets or voids

### 🧪 Testing

- Model evaluation scripts
- Dataset validation utilities
- Inference testing examples
- Performance benchmarking tools

### 📝 Known Limitations

- Requires labeled training data (no unsupervised learning)
- Performance depends on image quality and consistency
- Model trained on specific defect types (transfer may be needed for new defects)
- INT8 quantization may have slight accuracy degradation on edge cases

### 🚀 Future Roadmap

Planned features for upcoming releases:
- Defect localization with bounding boxes
- Grad-CAM visualization for explainability
- Multi-GPU training support
- Automatic hyperparameter tuning
- Web-based inference interface
- Multi-language documentation
- Pre-trained model zoo
- Federated learning support

---

## Version Format

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Change Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

---

## Unreleased

### In Development

- [ ] Defect localization (bounding boxes)
- [ ] Grad-CAM visualization
- [ ] Web UI for inference
- [ ] Expanded dataset support
- [ ] Additional model architectures

---

## How to Update This File

When contributing, please update this file with your changes:

1. Add your changes under `[Unreleased]` section
2. Use appropriate category (Added, Changed, Fixed, etc.)
3. Include issue/PR reference if applicable
4. Keep entries concise but descriptive

**Example**:
```markdown
### Added
- Defect localization using Grad-CAM (#42)
- REST API endpoint for batch processing (#45)

### Fixed
- Memory leak in batch inference (#43)
- Incorrect batch size handling on MPS device (#44)
```

---

## Release Links

- [1.0.0] - 2026-02-08 - Initial Release

---

## Contact

For questions about releases or to report issues:
- **GitHub Issues**: [https://github.com/adarshshkla/Fabgaurd-AI/issues](https://github.com/adarshshkla/Fabgaurd-AI/issues)
- **GitHub Discussions**: [https://github.com/adarshshkla/Fabgaurd-AI/discussions](https://github.com/adarshshkla/Fabgaurd-AI/discussions)

---

**Fabgaurd-AI** - *Guarding Silicon, One Die at a Time.* 🛡️

---

*Last Updated: February 2026*
