# 🏗️ System Architecture

## Table of Contents

- [Overview](#overview)
- [System Design](#system-design)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Knowledge Distillation](#knowledge-distillation)
- [Inference Pipeline](#inference-pipeline)
- [Deployment Strategy](#deployment-strategy)

---

## Overview

Fabgaurd-AI employs a sophisticated two-stage architecture that balances accuracy with computational efficiency, making it suitable for both cloud-based analysis and edge deployment in manufacturing environments.

### Design Principles

1. **Accuracy First**: Teacher model prioritizes detection accuracy
2. **Efficiency Second**: Student model optimized for real-time inference
3. **Modularity**: Independent training stages allow flexible optimization
4. **Scalability**: Architecture supports batch processing and real-time streaming
5. **Portability**: ONNX export enables cross-platform deployment

---

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     FABGAURD-AI SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────────┐
         │         DATA INGESTION LAYER                │
         │  • Image Loading                            │
         │  • Preprocessing & Normalization            │
         │  • Data Augmentation                        │
         └────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────────┐
         │      TEACHER MODEL TRAINING STAGE           │
         │                                             │
         │  EfficientNet-B0                            │
         │  ├── Feature Extraction                     │
         │  ├── Dropout Regularization                 │
         │  └── Classification Head                    │
         │                                             │
         │  Training: 30 epochs, AdamW, ReduceLROnPlateau │
         └────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────────┐
         │   KNOWLEDGE DISTILLATION STAGE              │
         │                                             │
         │  MobileNetV3-Small                          │
         │  ├── Lightweight Feature Extraction         │
         │  ├── Distillation Loss (T=4.0, α=0.7)      │
         │  └── Classification Head                    │
         │                                             │
         │  Training: Soft targets from teacher        │
         └────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────────┐
         │     QUANTIZATION & OPTIMIZATION             │
         │                                             │
         │  • INT8 Quantization                        │
         │  • ONNX Export                              │
         │  • Inference Optimization                   │
         └────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────────┐
         │         DEPLOYMENT OPTIONS                  │
         │                                             │
         │  ├── Cloud: FP32 Teacher Model             │
         │  ├── Edge: INT8 Student Model              │
         │  └── Mobile: ONNX Runtime                  │
         └────────────────────────────────────────────┘
```

---

## Model Architecture

### Teacher Model: EfficientNet-B0

**EfficientNet-B0** is chosen as the teacher model due to its excellent balance between accuracy and model size.

#### Architecture Details

```
Input: RGB Image (224×224×3)
    │
    ▼
┌─────────────────────────────────────┐
│     Stem Convolution                │
│     32 filters, 3×3, stride 2       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     MBConv Blocks (16 stages)       │
│     • Inverted Residuals            │
│     • Squeeze-Excitation            │
│     • Skip Connections              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Global Average Pooling          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Dropout (p=0.3)                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Dense Layer (1280 → 12)         │
└─────────────────────────────────────┘
    │
    ▼
Output: Logits for 12 classes
```

**Key Features**:
- **Compound Scaling**: Balanced scaling of depth, width, and resolution
- **MBConv Blocks**: Mobile inverted bottleneck convolution
- **Squeeze-Excitation**: Channel-wise attention mechanism
- **Parameters**: ~5.3 million
- **FLOPs**: ~0.39 billion

#### Custom Modifications

```python
# Original classifier
model.classifier[1] = nn.Linear(1280, 1000)

# Modified for Fabgaurd-AI
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),      # Increased dropout for regularization
    nn.Linear(1280, 12)      # 12 defect classes
)
```

### Student Model: MobileNetV3-Small

**MobileNetV3-Small** is the student model optimized for mobile and edge deployment.

#### Architecture Details

```
Input: RGB Image (224×224×3)
    │
    ▼
┌─────────────────────────────────────┐
│     Initial Convolution             │
│     16 filters, hard-swish          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Inverted Residual Blocks        │
│     • Depthwise Separable Conv      │
│     • SE Modules                    │
│     • h-swish Activation            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Global Average Pooling          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     Classifier (1024 → 12)          │
└─────────────────────────────────────┘
    │
    ▼
Output: Logits for 12 classes
```

**Key Features**:
- **Depthwise Separable Convolutions**: Reduced computational cost
- **h-swish Activation**: Computationally efficient activation
- **SE Modules**: Lightweight attention mechanism
- **Parameters**: ~2.5 million (~50% of teacher)
- **FLOPs**: ~0.06 billion (85% reduction)

---

## Training Pipeline

### Stage 1: Teacher Model Training

#### Data Preprocessing

```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Geometric Augmentations
    transforms.RandomAffine(
        degrees=10,              # ±10° rotation
        translate=(0.05, 0.05),  # 5% translation
        scale=(1.0, 1.2)         # 0-20% zoom
    ),
    
    # Photometric Augmentations
    transforms.ColorJitter(
        brightness=0.2,          # ±20% brightness
        contrast=0.2             # ±20% contrast
    ),
    
    # Flips
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    
    # Normalization
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )
])
```

**Augmentation Strategy**:
- **Gentle Zoom**: 1.2x maximum to preserve defect context
- **Moderate Contrast**: Helps distinguish subtle defects
- **Random Flips**: Defects have no preferred orientation
- **No Heavy Distortion**: Maintains defect morphology

#### Loss Function

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Label Smoothing**: Prevents overconfidence and improves generalization

#### Optimizer Configuration

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 regularization
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Maximize F1 score
    factor=0.5,        # Halve learning rate
    patience=2         # After 2 epochs without improvement
)
```

#### Class Imbalance Handling

```python
def get_weighted_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
```

**Weighted Sampling**: Ensures balanced training even with imbalanced datasets

### Stage 2: Knowledge Distillation

#### Distillation Loss Function

The core of knowledge distillation is the combined loss:

```python
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Soft Loss: Learn from teacher's soft targets
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    # Hard Loss: Learn from ground truth labels
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined Loss
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

**Parameters**:
- **Temperature (T)**: 4.0
  - Softens probability distributions
  - Reveals teacher's uncertainty and similarities between classes
- **Alpha (α)**: 0.7
  - 70% weight on soft targets (teacher knowledge)
  - 30% weight on hard targets (ground truth)

#### Mathematical Formulation

The distillation loss is computed as:

$$
\mathcal{L}_{distill} = \alpha \cdot \mathcal{L}_{soft} + (1-\alpha) \cdot \mathcal{L}_{hard}
$$

Where:

$$
\mathcal{L}_{soft} = T^2 \cdot \text{KL}\left(\frac{\text{softmax}(z_s/T)}{\text{softmax}(z_t/T)}\right)
$$

$$
\mathcal{L}_{hard} = \text{CrossEntropy}(z_s, y)
$$

- $z_s$: Student logits
- $z_t$: Teacher logits
- $y$: Ground truth labels
- $T$: Temperature
- $\alpha$: Balancing factor

#### Training Strategy

```python
# Freeze teacher model
teacher.eval()

for epoch in range(EPOCHS):
    for inputs, labels in train_loader:
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # Train student
        optimizer.zero_grad()
        student_logits = student(inputs)
        
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            T=4.0,
            alpha=0.7
        )
        
        loss.backward()
        optimizer.step()
```

**Key Points**:
1. Teacher model remains frozen (no gradient updates)
2. Student learns from both teacher and ground truth
3. Temperature controls knowledge transfer richness
4. Alpha balances mimicry vs. independent learning

---

## Inference Pipeline

### Preprocessing

```python
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
```

### Inference

```python
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class.item(), confidence
```

### Post-processing

```python
class_names = [
    'Bridge', 'Crack', 'Delamination', 'Gap',
    'Good', 'Open', 'Particle', 'Polishing',
    'Random', 'Short', 'VIAS', 'Void'
]

result = {
    'class': class_names[predicted_class],
    'confidence': confidence,
    'defect_detected': class_names[predicted_class] != 'Good'
}
```

---

## Deployment Strategy

### Cloud Deployment

**Use Case**: Batch processing, historical analysis

```python
# Load FP32 teacher model
model = load_teacher_model('teacher_b0_refined.pth')
model = model.to('cuda')
model.eval()

# High-accuracy inference
for batch in dataloader:
    predictions = model(batch)
    # Process predictions
```

**Advantages**:
- Maximum accuracy
- No deployment constraints
- Suitable for comprehensive analysis

### Edge Deployment

**Use Case**: Real-time inspection on manufacturing line

```python
# Load INT8 quantized student model
model = load_quantized_model('student_int8.onnx')

# Fast inference
def real_time_inference(image):
    preprocessed = preprocess(image)
    result = model.run(None, {'input': preprocessed})
    return postprocess(result)
```

**Advantages**:
- Low latency (~8ms)
- Reduced power consumption
- Small model size (~2.3 MB)

### Mobile Deployment

**Use Case**: Handheld inspection devices

```python
# ONNX Runtime Mobile
import onnxruntime as ort

session = ort.InferenceSession(
    'student_int8.onnx',
    providers=['CPUExecutionProvider']
)

def mobile_inference(image):
    input_tensor = preprocess(image)
    outputs = session.run(None, {
        session.get_inputs()[0].name: input_tensor
    })
    return postprocess(outputs[0])
```

---

## Performance Optimization

### Model Quantization

**INT8 Quantization Workflow**:

```python
import torch.quantization as quantization

# Post-training static quantization
model_fp32 = load_student_model()
model_fp32.eval()

# Fuse modules (conv + bn + relu)
model_fused = torch.quantization.fuse_modules(
    model_fp32,
    [['conv', 'bn', 'relu']]
)

# Prepare for quantization
model_prepared = torch.quantization.prepare(model_fused)

# Calibration with representative data
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# Convert to INT8
model_int8 = torch.quantization.convert(model_prepared)
```

**Benefits**:
- 4x model size reduction
- 2-4x inference speedup
- <2% accuracy loss

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'student_int8.onnx',
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
```

---

## Scalability Considerations

### Horizontal Scaling

```python
# Multi-GPU inference
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Distributed processing
for batch in distributed_dataloader:
    predictions = model(batch)
```

### Batch Processing

```python
# Optimal batch size for throughput
BATCH_SIZE = 64  # Adjust based on GPU memory

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)
```

### Caching Strategy

```python
# Model caching for faster startup
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    model = load_model()
    model.eval()
    return model
```

---

## Monitoring & Logging

### Performance Metrics

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.predictions = []
    
    def log_inference(self, start_time, end_time, prediction):
        self.inference_times.append(end_time - start_time)
        self.predictions.append(prediction)
    
    def get_stats(self):
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'throughput': len(self.predictions) / sum(self.inference_times)
        }
```

### Error Handling

```python
try:
    prediction = model(input_tensor)
except torch.cuda.OutOfMemory:
    # Fallback to CPU
    model = model.to('cpu')
    prediction = model(input_tensor.to('cpu'))
except Exception as e:
    # Log error and graceful degradation
    logger.error(f"Inference failed: {e}")
    return default_response
```

---

## Security Considerations

### Model Protection

- Models stored with access controls
- Encrypted model files for sensitive deployments
- Watermarking techniques for IP protection

### Input Validation

```python
def validate_input(image):
    # Check image dimensions
    if image.size[0] > 4096 or image.size[1] > 4096:
        raise ValueError("Image too large")
    
    # Check file format
    if image.format not in ['JPEG', 'PNG', 'BMP']:
        raise ValueError("Unsupported format")
    
    return image
```

---

## Future Architecture Enhancements

1. **Multi-model Ensemble**: Combine predictions from multiple models
2. **Attention Visualization**: Grad-CAM for explainable AI
3. **Online Learning**: Continuous model updates from production data
4. **AutoML Integration**: Automated hyperparameter tuning
5. **Federated Learning**: Privacy-preserving distributed training

---

## References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs
2. Howard, A., et al. (2019). Searching for MobileNetV3
3. Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network
4. Jacob, B., et al. (2018). Quantization and Training of Neural Networks

---

*Last Updated: February 2026*
