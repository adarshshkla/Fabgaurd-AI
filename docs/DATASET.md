# 📊 Dataset Guide

Comprehensive guide for preparing, organizing, and understanding the dataset for **Fabgaurd-AI**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Defect Categories](#defect-categories)
- [Data Collection Guidelines](#data-collection-guidelines)
- [Data Preprocessing](#data-preprocessing)
- [Data Augmentation](#data-augmentation)
- [Dataset Statistics](#dataset-statistics)
- [Quality Assurance](#quality-assurance)
- [Common Issues](#common-issues)

---

## Overview

The Fabgaurd-AI dataset consists of high-resolution images of semiconductor dies and PCB components, categorized into 12 distinct classes representing various manufacturing defects plus a "Good" class for defect-free samples.

### Dataset Requirements

- **Total Images**: Minimum 840 (70 per class), Recommended 3600+ (300+ per class)
- **Image Format**: JPEG, PNG, or BMP
- **Resolution**: Minimum 224×224 pixels, Recommended 512×512 or higher
- **Color Space**: RGB (3 channels)
- **File Size**: Varies (typically 50KB - 5MB per image)
- **Naming Convention**: Any alphanumeric naming is acceptable

---

## Dataset Structure

### Required Directory Organization

```
Fabgaurd-AI/
│
├── train/                      # Training set (70-80% of data)
│   ├── Bridge/                 # Bridge defect images
│   │   ├── bridge_001.jpg
│   │   ├── bridge_002.jpg
│   │   └── ...
│   ├── Crack/                  # Crack defect images
│   │   ├── crack_001.jpg
│   │   └── ...
│   ├── Delamination/           # Delamination defect images
│   ├── Gap/                    # Gap defect images
│   ├── Good/                   # Defect-free images
│   ├── Open/                   # Open circuit defect images
│   ├── Particle/               # Particle contamination images
│   ├── Polishing/              # Polishing defect images
│   ├── Random/                 # Random anomaly images
│   ├── Short/                  # Short circuit defect images
│   ├── VIAS/                   # VIAS defect images
│   └── Void/                   # Void defect images
│
├── val/                        # Validation set (10-15% of data)
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
│
└── test/                       # Test set (10-15% of data)
    ├── Bridge/
    ├── Crack/
    ├── Delamination/
    ├── Gap/
    ├── Good/
    ├── Open/
    ├── Particle/
    ├── Polishing/
    ├── Random/
    ├── Short/
    ├── VIAS/
    └── Void/
```

### Splitting Strategy

| Split | Percentage | Purpose | Min Samples | Recommended |
|-------|------------|---------|-------------|-------------|
| **Train** | 70-80% | Model training | 50 per class | 200+ per class |
| **Val** | 10-15% | Hyperparameter tuning | 10 per class | 50+ per class |
| **Test** | 10-15% | Final evaluation | 10 per class | 50+ per class |

**Example for 300 images per class**:
- Training: 210 images (70%)
- Validation: 45 images (15%)
- Test: 45 images (15%)

---

## Defect Categories

### 1. Bridge 🌉

**Description**: Unintended electrical connection between two or more conductors that should be isolated.

**Visual Characteristics**:
- Metallic connection spanning between traces
- Often appears as thin conductive filament
- Can be solder, copper, or other conductive material

**Severity**: **Critical** - Can cause short circuits

**Common Causes**:
- Excessive solder application
- Contamination during fabrication
- Inadequate trace spacing

**Example Scenarios**:
```
Normal:  ─────    ─────
Bridge:  ─────════─────  (unwanted connection)
```

---

### 2. Crack 💥

**Description**: Fracture or break in the substrate, solder mask, or conductive traces.

**Visual Characteristics**:
- Linear or branching discontinuity
- Often dark or light lines depending on material
- Can propagate across multiple layers

**Severity**: **Critical** - Structural and electrical failure

**Common Causes**:
- Mechanical stress
- Thermal cycling
- Manufacturing defects

---

### 3. Delamination 📄

**Description**: Separation of layers in multi-layer PCBs or packaging.

**Visual Characteristics**:
- Bubbling or lifting of layers
- Discoloration at edges
- Visible gaps between layers

**Severity**: **Critical** - Compromises electrical and mechanical integrity

**Common Causes**:
- Poor adhesion between layers
- Moisture absorption
- Thermal expansion mismatch
- Manufacturing process issues

---

### 4. Gap 📏

**Description**: Incomplete connection or missing material in traces or solder joints.

**Visual Characteristics**:
- Break in conductive path
- Missing trace segments
- Discontinuity in expected patterns

**Severity**: **Critical** - Open circuit condition

**Common Causes**:
- Under-etching during fabrication
- Solder wicking
- Mask misalignment

---

### 5. Good ✅

**Description**: Defect-free, properly manufactured component or die.

**Visual Characteristics**:
- Clean, uniform appearance
- No visible defects or anomalies
- Meets all quality specifications

**Severity**: **N/A** - Pass condition

**Note**: This class is crucial for training the model to distinguish between acceptable and defective products.

---

### 6. Open 🔓

**Description**: Discontinuity in an electrical path that should be continuous.

**Visual Characteristics**:
- Complete break in traces
- Missing vias or connections
- Visible separation in conductive paths

**Severity**: **Critical** - Circuit will not function

**Common Causes**:
- Manufacturing errors
- Over-etching
- Physical damage

---

### 7. Particle 🦠

**Description**: Foreign material or contamination on the surface.

**Visual Characteristics**:
- Visible foreign objects
- Discoloration or spots
- Irregular surface texture

**Severity**: **Medium to High** - Can cause shorts or reliability issues

**Common Causes**:
- Contaminated manufacturing environment
- Inadequate cleaning
- Airborne particles

---

### 8. Polishing 💎

**Description**: Surface finish irregularities or uneven polishing.

**Visual Characteristics**:
- Uneven reflectivity
- Scratches or swirl marks
- Inconsistent surface texture

**Severity**: **Low to Medium** - Primarily cosmetic, may affect bonding

**Common Causes**:
- Improper chemical-mechanical polishing (CMP)
- Worn polishing pads
- Contaminated slurry

---

### 9. Random 🎲

**Description**: Unclassified anomalies or defects that don't fit other categories.

**Visual Characteristics**:
- Varies widely
- May be combination of multiple issues
- Unusual or unexpected patterns

**Severity**: **Variable** - Depends on specific anomaly

**Note**: This class captures edge cases and helps model generalization.

---

### 10. Short ⚡

**Description**: Unintended low-resistance path between conductors.

**Visual Characteristics**:
- Similar to bridge but may be less visible
- Can be internal or surface-level
- Often requires electrical testing to confirm

**Severity**: **Critical** - Circuit malfunction

**Common Causes**:
- Solder bridges
- Conductive contamination
- Manufacturing defects

---

### 11. VIAS 🕳️

**Description**: Defects in vertical interconnect access (via) holes.

**Visual Characteristics**:
- Incomplete via filling
- Voids in via barrels
- Misaligned vias
- Broken vias

**Severity**: **Medium to High** - Can cause connection failures

**Common Causes**:
- Improper drilling
- Inadequate plating
- Etching issues

---

### 12. Void 🫧

**Description**: Air pockets or voids in solder joints, adhesives, or materials.

**Visual Characteristics**:
- Bubbles or empty spaces
- Lack of material where expected
- Darker regions in X-ray or visual inspection

**Severity**: **Medium to High** - Can reduce joint strength

**Common Causes**:
- Outgassing during reflow
- Trapped air or moisture
- Inadequate wetting

---

## Data Collection Guidelines

### Image Acquisition

#### Equipment Recommendations

| Inspection Method | Recommended For | Resolution |
|------------------|----------------|------------|
| **Optical Microscope** | Surface defects | 5MP+ |
| **Automated Optical Inspection (AOI)** | High-throughput | 2-5MP |
| **X-ray Inspection** | Internal defects (voids, vias) | 1-2MP |
| **Camera + Macro Lens** | Larger components | 12MP+ |

#### Capture Settings

```yaml
Lighting:
  - Type: Diffused white LED (preferred)
  - Angle: Top-down or 30-45° oblique
  - Intensity: Consistent across all images
  - Avoid: Harsh shadows, glare

Camera:
  - Focus: Manual, consistent depth of field
  - Exposure: Auto or fixed for consistency
  - White Balance: Fixed (not auto)
  - ISO: Lowest possible (100-400)

Position:
  - Distance: Fixed for all images
  - Angle: Perpendicular to surface
  - Centering: Defect in center of frame
```

### Image Quality Criteria

✅ **Good Quality**:
- In focus and sharp
- Evenly lit without shadows
- Defect clearly visible
- No motion blur
- Consistent background

❌ **Poor Quality**:
- Out of focus or blurry
- Over/under exposed
- Defect not visible
- Inconsistent lighting
- Motion artifacts

---

## Data Preprocessing

### Automatic Preprocessing (by Model)

The training scripts automatically apply:

```python
transforms.Resize((224, 224))          # Resize to model input
transforms.ToTensor()                   # Convert to tensor [0, 1]
transforms.Normalize(                   # Normalize to ImageNet stats
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Manual Preprocessing (Optional)

#### Cropping Region of Interest

```python
from PIL import Image

def crop_roi(image_path, bbox, output_path):
    """
    Crop region of interest from image.
    
    Args:
        image_path: Path to input image
        bbox: (left, top, right, bottom) coordinates
        output_path: Path to save cropped image
    """
    img = Image.open(image_path)
    cropped = img.crop(bbox)
    cropped.save(output_path)

# Example
crop_roi('raw_image.jpg', (100, 100, 400, 400), 'cropped.jpg')
```

#### Removing Extreme Outliers

```python
import cv2
import numpy as np

def remove_outliers(image_path):
    """Remove very bright or dark images."""
    img = cv2.imread(image_path)
    mean_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # Keep images with reasonable brightness (50-200 on 0-255 scale)
    return 50 <= mean_brightness <= 200
```

---

## Data Augmentation

### Training Augmentations (Automatic)

Applied during training to increase dataset diversity:

```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Geometric augmentations
    transforms.RandomAffine(
        degrees=10,                  # ±10° rotation
        translate=(0.05, 0.05),      # 5% translation
        scale=(1.0, 1.2)             # 0-20% zoom
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    
    # Color augmentations
    transforms.ColorJitter(
        brightness=0.2,              # ±20% brightness
        contrast=0.2                 # ±20% contrast
    ),
    
    # Normalization
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### Augmentation Rationale

| Augmentation | Purpose | Benefit |
|--------------|---------|---------|
| **Rotation** | Defects have no preferred orientation | Rotation invariance |
| **Translation** | Defect position varies | Position invariance |
| **Scaling** | Various zoom levels | Scale invariance |
| **Flips** | Symmetric appearance | Orientation invariance |
| **Color Jitter** | Lighting variations | Robustness to illumination |

### Advanced Augmentations (Optional)

For enhanced robustness:

```python
from torchvision import transforms

advanced_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Additional augmentations
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    
    transforms.RandomApply([
        transforms.RandomAdjustSharpness(sharpness_factor=2)
    ], p=0.3),
    
    # Standard augmentations
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1.0, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Dataset Statistics

### Recommended Distribution

For balanced training:

```
Total Images: 3600 (300 per class × 12 classes)

Per Class:
├── Train: 210 images (70%)
├── Val:   45 images (15%)
└── Test:  45 images (15%)

Defect Distribution:
├── Critical defects (8): 2400 images
│   ├── Bridge, Crack, Delamination, Gap
│   └── Open, Short, VIAS, Void
├── Medium defects (2): 600 images
│   ├── Particle, Random
├── Low severity (1): 300 images
│   └── Polishing
└── Good (1): 300 images
```

### Handling Class Imbalance

If your dataset is imbalanced:

**Option 1: Weighted Sampling** (Implemented):
```python
from torch.utils.data import WeightedRandomSampler

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

**Option 2: Oversampling Minority Classes**:
```python
from imblearn.over_sampling import RandomOverSampler

# Oversample rare classes to match majority class
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

**Option 3: Weighted Loss**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

---

## Quality Assurance

### Dataset Validation Script

```python
import os
from PIL import Image
from collections import Counter

def validate_dataset(root_dir):
    """Validate dataset structure and quality."""
    
    required_classes = [
        'Bridge', 'Crack', 'Delamination', 'Gap',
        'Good', 'Open', 'Particle', 'Polishing',
        'Random', 'Short', 'VIAS', 'Void'
    ]
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dir, split)
        
        if not os.path.exists(split_path):
            print(f"❌ Missing '{split}' directory")
            continue
        
        print(f"\n{'='*50}")
        print(f"Validating {split.upper()} set")
        print(f"{'='*50}")
        
        class_counts = {}
        corrupt_images = []
        
        for class_name in required_classes:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"  ❌ Missing class: {class_name}")
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            class_counts[class_name] = len(images)
            
            # Check for corrupt images
            for img_file in images:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception as e:
                    corrupt_images.append(img_path)
        
        # Print statistics
        print(f"\nClass Distribution:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            print(f"  {class_name:15s}: {count:4d} images")
        
        print(f"\nTotal Images: {sum(class_counts.values())}")
        print(f"Average per Class: {sum(class_counts.values()) / len(class_counts):.1f}")
        
        if corrupt_images:
            print(f"\n❌ Found {len(corrupt_images)} corrupt images:")
            for img in corrupt_images:
                print(f"  - {img}")
        else:
            print(f"\n✅ All images valid")

if __name__ == "__main__":
    validate_dataset(".")
```

Run validation:
```bash
python validate_dataset.py
```

---

## Common Issues

### Issue 1: Insufficient Data

**Problem**: < 50 images per class

**Solutions**:
1. Collect more data (preferred)
2. Use aggressive augmentation
3. Transfer learning from pretrained models
4. Consider synthetic data generation

### Issue 2: Class Imbalance

**Problem**: "Good" class has 1000 images, "Random" has 50

**Solutions**:
1. Use WeightedRandomSampler (implemented by default)
2. Oversample minority classes
3. Undersample majority classes
4. Use class-weighted loss function

### Issue 3: Image Quality Variations

**Problem**: Different lighting, cameras, zoom levels

**Solutions**:
1. Standardize capture process
2. Normalize brightness/contrast
3. Use strong augmentation during training
4. Consider training separate models per camera/setup

### Issue 4: Mislabeled Images

**Problem**: Images in wrong category

**Solutions**:
1. Manual review and correction
2. Train initial model and find high-confidence wrong predictions
3. Use clustering to identify outliers
4. Implement labeling quality checks

---

## Dataset Preparation Script

Complete script to organize your images:

```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split images into train/val/test sets.
    
    Args:
        source_dir: Directory containing class folders with images
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    classes = [d for d in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, d))]
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(f"{split}/{class_name}", exist_ok=True)
    
    # Split each class
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Copy files
        for img in train_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                f"train/{class_name}/{img}"
            )
        
        for img in val_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                f"val/{class_name}/{img}"
            )
        
        for img in test_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                f"test/{class_name}/{img}"
            )
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

if __name__ == "__main__":
    split_dataset("raw_data", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

---

## Next Steps

After preparing your dataset:

1. 📊 Run validation script to check dataset quality
2. 🔧 Review [USAGE.md](USAGE.md) for training instructions
3. 🏗️ Check [ARCHITECTURE.md](ARCHITECTURE.md) for system details
4. 🚀 Start training your models!

---

*Last Updated: February 2026*
