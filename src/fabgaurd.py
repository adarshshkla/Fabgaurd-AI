"""
Fabgaurd-AI Defect Detector API
Main module for inference and model deployment
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union
import os


class DefectDetector:
    """
    Fabgaurd-AI Defect Detection System
    
    A computer vision system for detecting manufacturing defects in semiconductor
    dies and PCB components.
    
    Example:
        >>> detector = DefectDetector('models/teacher_b0_refined.pth')
        >>> result = detector.predict('test/Crack/image.jpg')
        >>> print(f"Defect: {result['predicted_class']}, Confidence: {result['confidence']:.2%}")
    """
    
    CLASS_NAMES = [
        'Bridge', 'Crack', 'Delamination', 'Gap',
        'Good', 'Open', 'Particle', 'Polishing',
        'Random', 'Short', 'VIAS', 'Void'
    ]
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'auto',
        model_type: str = 'auto'
    ):
        """
        Initialize the DefectDetector.
        
        Args:
            model_path: Path to the model weights file (.pth)
            device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps')
            model_type: Model architecture ('auto', 'teacher', 'student')
        """
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
        
        print(f"Using device: {self.device}")
        
        # Auto-detect model type from filename
        if model_type == 'auto':
            if 'teacher' in model_path.lower():
                model_type = 'teacher'
            elif 'student' in model_path.lower():
                model_type = 'student'
            else:
                model_type = 'teacher'  # default
        
        self.model_type = model_type
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded: {model_type} ({self._count_parameters():,} parameters)")
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """Load PyTorch model from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_type == 'teacher':
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(num_ftrs, len(self.CLASS_NAMES))
            )
        else:  # student
            model = models.mobilenet_v3_small(weights=None)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(num_ftrs, len(self.CLASS_NAMES))
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except RuntimeError:
            # Try loading from checkpoint dictionary
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise
        
        return model
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def predict(
        self, 
        image_path: str, 
        return_probabilities: bool = False
    ) -> Dict[str, Union[str, float, bool, List, Dict]]:
        """
        Predict defect class for a single image.
        
        Args:
            image_path: Path to input image
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            Dictionary containing:
                - predicted_class: Name of predicted defect class
                - confidence: Confidence score [0, 1]
                - defect_detected: True if defect found, False if "Good"
                - top3: List of top 3 predictions [(class, prob), ...]
                - all_probabilities: Dict of all class probabilities (if requested)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top-3 predictions
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
    
    def predict_batch(
        self, 
        image_paths: List[str], 
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict defect classes for multiple images.
        
        Args:
            image_paths: List of paths to input images
            batch_size: Number of images to process at once
        
        Returns:
            List of prediction dictionaries (same format as predict())
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and transform batch
            batch_tensors = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = F.softmax(outputs, dim=1)
            
            # Process results
            for j in range(len(batch_tensors)):
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
    
    def predict_from_array(
        self, 
        image_array: np.ndarray, 
        return_probabilities: bool = False
    ) -> Dict:
        """
        Predict from numpy array (useful for integration with OpenCV).
        
        Args:
            image_array: Image as numpy array (H, W, 3), RGB format, uint8
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            Prediction dictionary (same format as predict())
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.CLASS_NAMES[predicted.item()]
        
        # Top-3 predictions
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
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information:
                - model_type: Architecture type ('teacher' or 'student')
                - num_parameters: Number of model parameters
                - device: Device model is running on
                - classes: List of class names
        """
        return {
            'model_type': self.model_type,
            'num_parameters': self._count_parameters(),
            'device': self.device,
            'classes': self.CLASS_NAMES
        }


def main():
    """Example usage of DefectDetector."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fabgaurd.py <model_path> <image_path>")
        print("Example: python fabgaurd.py models/teacher_b0_refined.pth test/Crack/image.jpg")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Initialize detector
    detector = DefectDetector(model_path)
    
    # Predict
    result = detector.predict(image_path, return_probabilities=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"IMAGE: {image_path}")
    print(f"{'='*60}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Defect Detected: {'YES' if result['defect_detected'] else 'NO'}")
    print(f"\nTop 3 Predictions:")
    for i, (cls, prob) in enumerate(result['top3'], 1):
        print(f"  {i}. {cls:15s} - {prob*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
