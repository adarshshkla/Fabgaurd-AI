# 📖 Usage Guide

Comprehensive guide for using **Fabgaurd-AI** to train, evaluate, and deploy defect detection models.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training Workflow](#training-workflow)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [Model Export](#model-export)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Running a Complete Training Pipeline

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 2. Navigate to source directory
cd src

# 3. Train teacher model (takes ~2-3 hours on GPU)
python train_teacher.py

# 4. Train student model (takes ~1-2 hours on GPU)
python train_student.py

# 5. Models will be saved in models/ directory
```

---

## Dataset Preparation

### Required Directory Structure

Your dataset must follow this structure:

```
Fabgaurd-AI/
├── train/
│   ├── Bridge/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
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

### Dataset Split Recommendations

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Train** | 70-80% | Model training |
| **Validation** | 10-15% | Hyperparameter tuning, model selection |
| **Test** | 10-15% | Final model evaluation |

### Example: Organizing Your Images

```bash
# Create directory structure
mkdir -p train val test

for dir in Bridge Crack Delamination Gap Good Open Particle Polishing Random Short VIAS Void; do
    mkdir -p train/$dir val/$dir test/$dir
done

# Move or copy your images to appropriate folders
# Example:
cp /path/to/bridge_defects/*.jpg train/Bridge/
cp /path/to/crack_defects/*.jpg train/Crack/
# ... and so on
```

### Image Requirements

- **Format**: JPEG, PNG, BMP
- **Resolution**: Minimum 224x224 pixels (will be resized)
- **Color**: RGB (3 channels)
- **Quality**: High quality, in-focus images recommended
- **Naming**: Any naming convention (alphanumeric recommended)

### Minimum Samples per Class

| Dataset Size | Train | Val | Test |
|--------------|-------|-----|------|
| **Minimum** | 50 | 10 | 10 |
| **Recommended** | 200+ | 50+ | 50+ |
| **Ideal** | 1000+ | 200+ | 200+ |

**Note**: More samples generally lead to better model performance.

---

## Training Workflow

### Stage 1: Training the Teacher Model

#### Configuration

Edit `src/train_teacher.py` to configure training:

```python
# Training hyperparameters
BATCH_SIZE = 32         # Reduce if GPU memory is limited
EPOCHS = 30             # Number of training epochs
LEARNING_RATE = 1e-3    # Initial learning rate

# Data paths (auto-detected by default)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = "teacher_b0_refined.pth"
```

#### Run Training

```bash
cd src
python train_teacher.py
```

#### Training Output

```
🔬 Launching REFINED Teacher (EfficientNet-B0) on cuda...
🧠 Loading EfficientNet-B0...

   Epoch 1: Batch 50/100...
✅ Epoch 1/30 | F1: 75.34% | LR: 1.00e-03
   💾 Saved Best Refined Model!

   Epoch 2: Batch 48/100...
✅ Epoch 2/30 | F1: 82.19% | LR: 1.00e-03
   💾 Saved Best Refined Model!

              precision    recall  f1-score   support

      Bridge       0.89      0.85      0.87        40
       Crack       0.92      0.88      0.90        50
      ...

🏆 Done in 142m 35s
📁 Saved to: teacher_b0_refined.pth
```

#### Key Metrics to Monitor

- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Learning Rate**: Should decrease when validation plateaus
- **Loss**: Should decrease steadily

#### When to Stop

The training automatically:
- Saves the best model based on F1-score
- Reduces learning rate when improvement plateaus
- Can be stopped early if convergence is reached

**Manual Early Stopping**: Press `Ctrl+C` to stop (best model will be saved)

---

### Stage 2: Training the Student Model (Distillation)

#### Prerequisites

Ensure teacher model exists:
```bash
ls -lh ../models/teacher_b0_refined.pth
```

#### Configuration

Edit `src/train_student.py`:

```python
# Model paths
TEACHER_PATH = "../models/teacher_b0_refined.pth"
STUDENT_PATH = "../models/student_mobilenet_small.pth"

# Distillation hyperparameters
TEMPERATURE = 4.0     # Temperature for soft targets (higher = softer)
ALPHA = 0.7           # Weight for soft loss (0.7 = 70% teacher, 30% labels)

# Training settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
```

#### Understanding Hyperparameters

**Temperature (T)**:
- Higher (e.g., 6.0): More knowledge transfer, smoother learning
- Lower (e.g., 2.0): Harder targets, faster convergence
- **Recommended**: 4.0

**Alpha (α)**:
- Higher (e.g., 0.9): Student relies more on teacher
- Lower (e.g., 0.5): Student relies more on ground truth
- **Recommended**: 0.7

#### Run Distillation

```bash
python train_student.py
```

#### Training Output

```
🎓 Launching Tiny Student (MobileNetV3-Small) on cuda...
👨‍🏫 Loading Teacher...
👶 Creating Tiny Student...

   Epoch 1: Batch 50/100...
✅ Epoch 1/30 | Tiny Student F1: 70.45%
   💾 Saved Best Tiny Model! (student_mobilenet_small.pth)

   Epoch 2: Batch 50/100...
✅ Epoch 2/30 | Tiny Student F1: 78.32%
   💾 Saved Best Tiny Model! (student_mobilenet_small.pth)

🏆 Distillation Complete in 85m 22s
📁 Final Tiny Model: student_mobilenet_small.pth
```

#### Expected Performance

| Model | F1-Score | Training Time |
|-------|----------|---------------|
| Teacher (EfficientNet-B0) | ~94% | 2-3 hours |
| Student (MobileNetV3) | ~90% | 1-2 hours |
| Performance Gap | ~4% | - |

---

## Model Evaluation

### Evaluate on Test Set

Create `src/evaluate.py`:

```python
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, data_dir):
    # Load model
    model = models.efficientnet_b0(weights=None)
    num_classes = 12
    model.classifier[1] = torch.nn.Linear(1280, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare test data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(
        f"{data_dir}/test",
        transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Inference
    all_preds = []
    all_labels = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print results
    class_names = test_dataset.classes
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    return all_preds, all_labels

if __name__ == "__main__":
    evaluate_model("../models/teacher_b0_refined.pth", "..")
```

Run evaluation:
```bash
python evaluate.py
```

### Interpreting Results

#### Classification Report

```
              precision    recall  f1-score   support

      Bridge       0.92      0.88      0.90        40
       Crack       0.95      0.91      0.93        50
 Delamination       0.89      0.93      0.91        45
         Gap       0.87      0.85      0.86        38
        Good       0.98      0.97      0.97       100
        Open       0.91      0.89      0.90        42
    Particle       0.85      0.83      0.84        35
   Polishing       0.82      0.86      0.84        30
      Random       0.79      0.75      0.77        28
       Short       0.88      0.92      0.90        45
        VIAS       0.86      0.84      0.85        32
        Void       0.90      0.88      0.89        40

    accuracy                           0.89       525
   macro avg       0.89      0.88      0.88       525
weighted avg       0.90      0.89      0.89       525
```

**Key Metrics**:
- **Precision**: Of predicted defects, how many are actually defects?
- **Recall**: Of actual defects, how many did we catch?
- **F1-Score**: Balanced metric (harmonic mean)
- **Support**: Number of samples per class

#### Confusion Matrix Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

# After evaluation
plot_confusion_matrix(cm, class_names)
```

---

## Inference

### Single Image Inference

Create `src/predict.py`:

```python
import torch
from torchvision import models, transforms
from PIL import Image
import sys

class DefectDetector:
    def __init__(self, model_path, device='auto'):
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = models.efficientnet_b0(weights=None)
        num_classes = 12
        self.model.classifier[1] = torch.nn.Linear(1280, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = [
            'Bridge', 'Crack', 'Delamination', 'Gap',
            'Good', 'Open', 'Particle', 'Polishing',
            'Random', 'Short', 'VIAS', 'Void'
        ]
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Results
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top-3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_classes = [(self.class_names[idx], prob.item()) 
                       for idx, prob in zip(top3_idx[0], top3_prob[0])]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'defect_detected': predicted_class != 'Good',
            'top3': top3_classes
        }
    
    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            result = self.predict(path)
            results.append(result)
        return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize detector
    detector = DefectDetector('../models/teacher_b0_refined.pth')
    
    # Predict
    result = detector.predict(image_path)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"IMAGE: {image_path}")
    print(f"{'='*50}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Defect Detected: {'YES' if result['defect_detected'] else 'NO'}")
    print(f"\nTop 3 Predictions:")
    for i, (cls, prob) in enumerate(result['top3'], 1):
        print(f"  {i}. {cls:15s} - {prob*100:.2f}%")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
```

#### Usage

```bash
# Single image
python predict.py /path/to/test/image.jpg

# Output:
# ==================================================
# IMAGE: /path/to/test/image.jpg
# ==================================================
# Predicted Class: Crack
# Confidence: 95.34%
# Defect Detected: YES
#
# Top 3 Predictions:
#   1. Crack           - 95.34%
#   2. Delamination    - 3.12%
#   3. Gap             - 1.05%
# ==================================================
```

### Batch Inference

```python
# Batch processing
image_paths = [
    'test/Crack/img001.jpg',
    'test/Bridge/img002.jpg',
    'test/Good/img003.jpg'
]

detector = DefectDetector('../models/teacher_b0_refined.pth')
results = detector.predict_batch(image_paths)

for path, result in zip(image_paths, results):
    print(f"{path}: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
```

---

## Model Export

### Export to ONNX

Create `src/export_onnx.py`:

```python
import torch
from torchvision import models

def export_to_onnx(pytorch_model_path, onnx_model_path):
    # Load PyTorch model
    model = models.mobilenet_v3_small(weights=None)
    num_classes = 12
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to {onnx_model_path}")

if __name__ == "__main__":
    export_to_onnx(
        '../models/student_mobilenet_small.pth',
        '../models/student_mobilenet_small.onnx'
    )
```

Run export:
```bash
python export_onnx.py
```

### Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

def predict_onnx(onnx_model_path, image_path):
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).numpy()
    
    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    
    # Post-process
    probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[0, predicted_class]
    
    class_names = [
        'Bridge', 'Crack', 'Delamination', 'Gap',
        'Good', 'Open', 'Particle', 'Polishing',
        'Random', 'Short', 'VIAS', 'Void'
    ]
    
    return class_names[predicted_class], confidence

# Usage
predicted, confidence = predict_onnx(
    '../models/student_mobilenet_small.onnx',
    'test/Crack/image.jpg'
)
print(f"Predicted: {predicted} ({confidence*100:.2f}%)")
```

---

## Best Practices

### Training Tips

1. **Start with default hyperparameters**: The provided settings work well for most cases

2. **Monitor overfitting**:
   - If training F1 >> validation F1, you're overfitting
   - Solutions: Increase dropout, add more augmentation, reduce epochs

3. **Learning rate tuning**:
   - Too high: Loss oscillates
   - Too low: Slow convergence
   - Default 1e-3 is usually good

4. **Batch size selection**:
   - GPU memory limited: Reduce batch size
   - Larger batches: More stable training
   - Smaller batches: Better generalization

5. **Data augmentation balance**:
   - Too aggressive: Destroys defect features
   - Too mild: Underfitting
   - Current settings are balanced

### Inference Optimization

```python
# 1. Use torch.no_grad()
with torch.no_grad():
    outputs = model(inputs)

# 2. Set model to eval mode
model.eval()

# 3. Use smaller model for production
# Student model is 3x faster than teacher

# 4. Batch processing for multiple images
# Process in batches of 16-32 for best throughput

# 5. Use appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Memory Management

```python
# Clear CUDA cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use mixed precision (FP16) for training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Advanced Usage

### Custom Data Augmentation

```python
from torchvision import transforms
from PIL import ImageFilter

class GaussianBlur:
    def __init__(self, radius):
        self.radius = radius
    
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

custom_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([GaussianBlur(2)], p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Learning Rate Scheduling

```python
# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# Step LR
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# One Cycle LR
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)
```

### Model Ensemble

```python
def ensemble_predict(models, image):
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs)
    
    # Average predictions
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    predicted_class = torch.argmax(ensemble_pred, dim=1)
    
    return predicted_class

# Usage
models = [teacher_model, student_model]
result = ensemble_predict(models, input_tensor)
```

---

## Troubleshooting

### Common Issues During Training

**Issue**: Training loss not decreasing
- Check learning rate (try 1e-4 or 1e-2)
- Verify data loading (print batch shapes)
- Check if model is frozen

**Issue**: Validation F1 much lower than training
- Increase dropout (try 0.5)
- Reduce model complexity
- Add more training data

**Issue**: Out of memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Common Issues During Inference

**Issue**: Wrong predictions
- Verify image preprocessing matches training
- Check model path is correct
- Ensure class order matches

**Issue**: Slow inference
- Use student model instead of teacher
- Batch multiple images
- Enable CUDA if available

---

## Next Steps

- 📊 Review [MODEL_DETAILS.md](MODEL_DETAILS.md) for architecture details
- 📁 Check [DATASET.md](DATASET.md) for dataset guidelines
- 🔌 See [API_REFERENCE.md](API_REFERENCE.md) for integration

---

*Last Updated: February 2026*
