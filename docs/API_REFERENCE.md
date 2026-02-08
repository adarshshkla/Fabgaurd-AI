# 🔌 API Reference

Complete API documentation for **Fabgaurd-AI** model inference and integration.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Python API](#python-api)
- [REST API](#rest-api-optional)
- [ONNX Runtime API](#onnx-runtime-api)
- [Batch Processing](#batch-processing)
- [Integration Examples](#integration-examples)
- [Error Handling](#error-handling)

---

## Quick Start

### Basic Inference

```python
from fabgaurd import DefectDetector

# Initialize detector
detector = DefectDetector(model_path='models/teacher_b0_refined.pth')

# Predict single image
result = detector.predict('path/to/image.jpg')

print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Defect: {result['defect_detected']}")
```

---

## Python API

### DefectDetector Class

Main class for defect detection inference.

#### Constructor

```python
class DefectDetector:
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        model_type: str = 'teacher'
    )
```

**Parameters**:
- `model_path` (str): Path to the trained model file (.pth or .onnx)
- `device` (str, optional): Device to run inference on. Options:
  - `'auto'`: Automatically select best available device (default)
  - `'cuda'`: Use NVIDIA GPU
  - `'cpu'`: Use CPU
  - `'mps'`: Use Apple Silicon GPU
- `model_type` (str, optional): Model architecture type
  - `'teacher'`: EfficientNet-B0 (default if filename contains 'teacher')
  - `'student'`: MobileNetV3-Small (default if filename contains 'student')

**Example**:
```python
# Auto-detect device
detector = DefectDetector('models/teacher_b0_refined.pth')

# Force CPU
detector = DefectDetector('models/student_mobilenet_small.pth', device='cpu')

# Explicit model type
detector = DefectDetector('custom_model.pth', model_type='student')
```

#### Methods

##### `predict()`

Predict defect class for a single image.

```python
def predict(
    self,
    image_path: str,
    return_probabilities: bool = False
) -> dict
```

**Parameters**:
- `image_path` (str): Path to the input image
- `return_probabilities` (bool, optional): Return probabilities for all classes (default: False)

**Returns**:
- `dict`: Prediction results with keys:
  - `predicted_class` (str): Name of predicted class
  - `confidence` (float): Confidence score [0.0, 1.0]
  - `defect_detected` (bool): True if defect found, False if "Good"
  - `top3` (list): Top 3 predictions [(class, probability), ...]
  - `all_probabilities` (dict, optional): Probabilities for all classes

**Example**:
```python
result = detector.predict('test/Crack/image001.jpg')
# {
#     'predicted_class': 'Crack',
#     'confidence': 0.9534,
#     'defect_detected': True,
#     'top3': [
#         ('Crack', 0.9534),
#         ('Delamination', 0.0312),
#         ('Gap', 0.0105)
#     ]
# }

# With all probabilities
result = detector.predict('test/image.jpg', return_probabilities=True)
# Includes 'all_probabilities': {'Bridge': 0.001, 'Crack': 0.953, ...}
```

##### `predict_batch()`

Predict defect classes for multiple images.

```python
def predict_batch(
    self,
    image_paths: list[str],
    batch_size: int = 32
) -> list[dict]
```

**Parameters**:
- `image_paths` (list[str]): List of input image paths
- `batch_size` (int, optional): Batch size for inference (default: 32)

**Returns**:
- `list[dict]`: List of prediction results (same format as `predict()`)

**Example**:
```python
image_paths = [
    'test/Crack/img1.jpg',
    'test/Bridge/img2.jpg',
    'test/Good/img3.jpg'
]

results = detector.predict_batch(image_paths, batch_size=16)

for path, result in zip(image_paths, results):
    print(f"{path}: {result['predicted_class']} ({result['confidence']:.2%})")
```

##### `predict_from_array()`

Predict from a numpy array (useful for integration with other vision systems).

```python
def predict_from_array(
    self,
    image_array: np.ndarray,
    return_probabilities: bool = False
) -> dict
```

**Parameters**:
- `image_array` (np.ndarray): Image as numpy array (H, W, 3), RGB format, uint8
- `return_probabilities` (bool, optional): Return all class probabilities

**Returns**:
- `dict`: Prediction results (same format as `predict()`)

**Example**:
```python
import cv2
import numpy as np

# Read image with OpenCV (BGR)
image_bgr = cv2.imread('test/image.jpg')

# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Predict
result = detector.predict_from_array(image_rgb)
```

##### `get_model_info()`

Get information about the loaded model.

```python
def get_model_info(self) -> dict
```

**Returns**:
- `dict`: Model information with keys:
  - `model_type` (str): Architecture type ('teacher' or 'student')
  - `num_parameters` (int): Number of model parameters
  - `device` (str): Device model is running on
  - `classes` (list): List of class names

**Example**:
```python
info = detector.get_model_info()
print(f"Model: {info['model_type']}")
print(f"Parameters: {info['num_parameters']:,}")
print(f"Device: {info['device']}")
print(f"Classes: {', '.join(info['classes'])}")
```

---

### Complete Implementation Example

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

class DefectDetector:
    """Fabgaurd-AI Defect Detector"""
    
    CLASS_NAMES = [
        'Bridge', 'Crack', 'Delamination', 'Gap',
        'Good', 'Open', 'Particle', 'Polishing',
        'Random', 'Short', 'VIAS', 'Void'
    ]
    
    def __init__(self, model_path, device='auto', model_type='auto'):
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
        
        # Auto-detect model type
        if model_type == 'auto':
            if 'teacher' in model_path.lower():
                model_type = 'teacher'
            elif 'student' in model_path.lower():
                model_type = 'student'
            else:
                model_type = 'teacher'  # default
        
        self.model_type = model_type
        
        # Load model
        if model_type == 'teacher':
            self.model = models.efficientnet_b0(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(num_ftrs, len(self.CLASS_NAMES))
            )
        else:  # student
            self.model = models.mobilenet_v3_small(weights=None)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = torch.nn.Linear(num_ftrs, len(self.CLASS_NAMES))
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, return_probabilities=False):
        """Predict single image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Top-3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_classes = [
            (self.CLASS_NAMES[idx], prob.item())
            for idx, prob in zip(top3_idx[0], top3_prob[0])
        ]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'defect_detected': predicted_class != 'Good',
            'top3': top3_classes
        }
        
        if return_probabilities:
            result['all_probabilities'] = {
                cls: probabilities[0, i].item()
                for i, cls in enumerate(self.CLASS_NAMES)
            }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict batch of images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and transform batch
            batch_tensors = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = F.softmax(outputs, dim=1)
            
            # Process results
            for j in range(len(batch_paths)):
                confidence, predicted = torch.max(probabilities[j], 0)
                predicted_class = self.CLASS_NAMES[predicted.item()]
                
                top3_prob, top3_idx = torch.topk(probabilities[j], 3)
                top3_classes = [
                    (self.CLASS_NAMES[idx], prob.item())
                    for idx, prob in zip(top3_idx, top3_prob)
                ]
                
                results.append({
                    'predicted_class': predicted_class,
                    'confidence': confidence.item(),
                    'defect_detected': predicted_class != 'Good',
                    'top3': top3_classes
                })
        
        return results
    
    def predict_from_array(self, image_array, return_probabilities=False):
        """Predict from numpy array"""
        image = Image.fromarray(image_array)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.CLASS_NAMES[predicted.item()]
        
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_classes = [
            (self.CLASS_NAMES[idx], prob.item())
            for idx, prob in zip(top3_idx[0], top3_prob[0])
        ]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence.item(),
            'defect_detected': predicted_class != 'Good',
            'top3': top3_classes
        }
        
        if return_probabilities:
            result['all_probabilities'] = {
                cls: probabilities[0, i].item()
                for i, cls in enumerate(self.CLASS_NAMES)
            }
        
        return result
    
    def get_model_info(self):
        """Get model information"""
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'model_type': self.model_type,
            'num_parameters': num_params,
            'device': self.device,
            'classes': self.CLASS_NAMES
        }
```

**Save as `src/fabgaurd.py`** and import:
```python
from fabgaurd import DefectDetector
```

---

## REST API (Optional)

### Flask-based REST API

Create a simple REST API for remote inference:

```python
# api_server.py
from flask import Flask, request, jsonify
from fabgaurd import DefectDetector
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize detector
detector = DefectDetector('models/student_mobilenet_small.pth')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict defect class for uploaded image.
    
    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image file
    
    Response:
        {
            "predicted_class": "Crack",
            "confidence": 0.9534,
            "defect_detected": true,
            "top3": [["Crack", 0.9534], ...]
        }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)
    
    try:
        # Predict
        result = detector.predict(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result), 200
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': detector.get_model_info()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Run server**:
```bash
python api_server.py
```

**Client usage**:
```python
import requests

# Upload and predict
with open('test/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )

result = response.json()
print(result['predicted_class'])
```

**cURL example**:
```bash
curl -X POST -F "image=@test/image.jpg" http://localhost:5000/predict
```

---

## ONNX Runtime API

For cross-platform deployment using ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

class ONNXDefectDetector:
    """ONNX Runtime-based detector"""
    
    CLASS_NAMES = [
        'Bridge', 'Crack', 'Delamination', 'Gap',
        'Good', 'Open', 'Particle', 'Polishing',
        'Random', 'Short', 'VIAS', 'Void'
    ]
    
    def __init__(self, onnx_model_path):
        # Create inference session
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict using ONNX Runtime"""
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).numpy()
        
        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Top prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Top 3
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_classes = [(self.CLASS_NAMES[i], probabilities[i]) for i in top3_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'defect_detected': predicted_class != 'Good',
            'top3': top3_classes
        }

# Usage
detector = ONNXDefectDetector('models/student_mobilenet_small.onnx')
result = detector.predict('test/image.jpg')
```

---

## Batch Processing

### Process Directory of Images

```python
import os
from pathlib import Path
from fabgaurd import DefectDetector

def process_directory(input_dir, output_csv='results.csv'):
    """Process all images in directory"""
    detector = DefectDetector('models/teacher_b0_refined.pth')
    
    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(Path(input_dir).rglob(ext))
    
    image_paths = [str(p) for p in image_paths]
    
    print(f"Found {len(image_paths)} images")
    
    # Batch predict
    results = detector.predict_batch(image_paths, batch_size=32)
    
    # Save to CSV
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted Class', 'Confidence', 'Defect Detected'])
        
        for path, result in zip(image_paths, results):
            writer.writerow([
                path,
                result['predicted_class'],
                f"{result['confidence']:.4f}",
                result['defect_detected']
            ])
    
    print(f"Results saved to {output_csv}")
    
    # Summary
    defects_found = sum(1 for r in results if r['defect_detected'])
    print(f"\nSummary:")
    print(f"  Total images: {len(results)}")
    print(f"  Defects found: {defects_found}")
    print(f"  Good: {len(results) - defects_found}")

# Usage
process_directory('test/', output_csv='inspection_results.csv')
```

---

## Integration Examples

### OpenCV Integration

```python
import cv2
from fabgaurd import DefectDetector

detector = DefectDetector('models/student_mobilenet_small.pth')

# Read with OpenCV
image_bgr = cv2.imread('test/image.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Predict
result = detector.predict_from_array(image_rgb)

# Draw result on image
color = (0, 0, 255) if result['defect_detected'] else (0, 255, 0)
cv2.putText(
    image_bgr,
    f"{result['predicted_class']}: {result['confidence']:.2%}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    color,
    2
)

cv2.imshow('Result', image_bgr)
cv2.waitKey(0)
```

### Real-time Video Processing

```python
import cv2
from fabgaurd import DefectDetector

detector = DefectDetector('models/student_mobilenet_small.pth')

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Predict
    result = detector.predict_from_array(frame_rgb)
    
    # Display result
    color = (0, 0, 255) if result['defect_detected'] else (0, 255, 0)
    cv2.putText(
        frame,
        f"{result['predicted_class']}: {result['confidence']:.1%}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        color,
        3
    )
    
    cv2.imshow('Defect Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Error Handling

### Robust Error Handling

```python
from fabgaurd import DefectDetector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    detector = DefectDetector('models/teacher_b0_refined.pth')
except FileNotFoundError:
    logger.error("Model file not found!")
    exit(1)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

def safe_predict(image_path):
    """Predict with error handling"""
    try:
        result = detector.predict(image_path)
        return result
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Prediction failed for {image_path}: {e}")
        return None

# Usage
result = safe_predict('test/image.jpg')
if result:
    print(f"Prediction: {result['predicted_class']}")
else:
    print("Prediction failed")
```

---

## Performance Tips

1. **Use Student Model for Production**: 3x faster than teacher
2. **Batch Processing**: Process multiple images at once for better throughput
3. **GPU Acceleration**: Use CUDA if available
4. **INT8 Quantization**: For edge deployment
5. **ONNX Runtime**: For cross-platform compatibility

---

*Last Updated: February 2026*
