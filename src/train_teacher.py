import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(DATA_DIR, "models", "teacher_b0_refined.pth")

# SETTINGS
BATCH_SIZE = 32      
EPOCHS = 30          
LEARNING_RATE = 1e-3 

def get_device_config():
    if torch.cuda.is_available():
        return "cuda", 0, True 
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", 0, False 
    return "cpu", 0, False

DEVICE, NUM_WORKERS, PIN_MEMORY = get_device_config()

def get_weighted_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def train_teacher_refined():
    print(f"\n🔬 Launching REFINED Teacher (EfficientNet-B0) on {DEVICE}...")
    start_time = time.time()

    # 1. REFINED TRANSFORMS (The "Goldilocks" Zone)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # --- FIX: Gentler Zoom (1.2x instead of 1.5x) ---
        # This keeps 'Bridge' context visible while still catching 'Gaps'.
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1.0, 1.2)), 
        
        # Moderate Contrast (Helps 'Good' vs 'Short')
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. DATASETS
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_dir):
        print("❌ Error: 'train' folder not found!")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    class_names = train_dataset.classes

    # 3. DATALOADERS
    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # 4. MODEL (EfficientNet-B0)
    print("🧠 Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3), 
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(DEVICE)

    # 5. OPTIMIZER & SCHEDULER
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- FIX: Removed 'verbose=True' to fix error ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # 6. TRAINING LOOP
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f"   Epoch {epoch+1}: Batch {i+1}/{len(train_loader)}...", end='\r')

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        curr_f1 = report['macro avg']['f1-score']
        curr_lr = optimizer.param_groups[0]['lr']
        
        # Step the scheduler based on F1 Score
        scheduler.step(curr_f1)
        
        print(f"\n✅ Epoch {epoch+1}/{EPOCHS} | F1: {curr_f1:.2%} | LR: {curr_lr:.2e}")

        if curr_f1 > best_f1:
            best_f1 = curr_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   💾 Saved Best Refined Model!")
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    elapsed = time.time() - start_time
    print(f"\n🏆 Done in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"📁 Saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_teacher_refined()