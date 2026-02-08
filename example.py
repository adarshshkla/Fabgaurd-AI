"""
Simple example demonstrating Fabgaurd-AI usage
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from fabgaurd import DefectDetector


def main():
    """
    Example demonstrating basic usage of Fabgaurd-AI
    """
    print("="*60)
    print("Fabgaurd-AI - Defect Detection Example")
    print("="*60)
    
    # Initialize detector with teacher model
    print("\n1. Loading model...")
    detector = DefectDetector('models/teacher_b0_refined.pth')
    
    # Get model information
    info = detector.get_model_info()
    print(f"\nModel Info:")
    print(f"  Type: {info['model_type']}")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Device: {info['device']}")
    print(f"  Classes: {len(info['classes'])}")
    
    # Example: Predict single image
    print("\n2. Predicting single image...")
    
    # Find a test image
    test_dirs = ['test', 'val']
    image_path = None
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for class_dir in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_dir)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if images:
                        image_path = os.path.join(class_path, images[0])
                        break
            if image_path:
                break
    
    if image_path:
        result = detector.predict(image_path)
        
        print(f"\nImage: {image_path}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Defect: {'YES' if result['defect_detected'] else 'NO'}")
        print(f"\nTop 3 Predictions:")
        for i, (cls, prob) in enumerate(result['top3'], 1):
            print(f"  {i}. {cls:15s} - {prob*100:.2f}%")
    else:
        print("No test images found. Please organize your dataset first.")
        print("See docs/DATASET.md for dataset structure information.")
    
    # Example: Batch prediction
    print("\n3. Batch prediction example...")
    
    # Find multiple test images
    image_paths = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for class_dir in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_dir)
                if os.path.isdir(class_path):
                    images = [os.path.join(class_path, f) 
                             for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    image_paths.extend(images[:2])  # Take 2 images per class
                    if len(image_paths) >= 10:
                        break
            if len(image_paths) >= 10:
                break
    
    if image_paths:
        print(f"Processing {len(image_paths)} images...")
        results = detector.predict_batch(image_paths, batch_size=16)
        
        # Summary
        defects_found = sum(1 for r in results if r['defect_detected'])
        print(f"\nBatch Results:")
        print(f"  Total images: {len(results)}")
        print(f"  Defects found: {defects_found}")
        print(f"  Good: {len(results) - defects_found}")
        
        # Show first few results
        print(f"\nFirst 3 results:")
        for i, (path, result) in enumerate(zip(image_paths[:3], results[:3])):
            status = "❌" if result['defect_detected'] else "✅"
            print(f"  {status} {os.path.basename(path)}: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nNext steps:")
    print("  • Train your own models: python src/train_teacher.py")
    print("  • Read documentation: docs/USAGE.md")
    print("  • Explore API: docs/API_REFERENCE.md")
    print()


if __name__ == "__main__":
    main()
