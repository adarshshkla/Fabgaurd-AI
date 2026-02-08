import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Small_Weights # <--- CHANGED
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_PATH = os.path.join(DATA_DIR, "models", "teacher_b0_refined.pth")
STUDENT_PATH = os.path.join(DATA_DIR, "models", "student_mobilenet_small.pth") # <--- NEW FILENAME

# HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
TEMPERATURE = 4.0
ALPHA = 0.7 

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

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss

def train_student():
    print(f"\n🎓 Launching Tiny Student (MobileNetV3-Small) on {DEVICE}...")
    start_time = time.time()

    # 1. DATA SETUP
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1.0, 1.2)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    
    if not os.path.exists(TEACHER_PATH):
        print(f"❌ Error: Teacher model '{TEACHER_PATH}' not found!")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. LOAD TEACHER
    print("👨‍🏫 Loading Teacher...")
    teacher = models.efficientnet_b0(weights=None)
    
    num_ftrs = teacher.classifier[1].in_features 
    teacher.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3), 
        nn.Linear(num_ftrs, num_classes)
    )

    try:
        teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    except RuntimeError:
        print("⚠️ Warning: Direct load failed, trying dictionary load...")
        checkpoint = torch.load(TEACHER_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['model_state_dict'])
        else:
            teacher.load_state_dict(checkpoint)
        
    teacher.to(DEVICE)
    teacher.eval() 

    # 3. BUILD TINY STUDENT (MobileNet V3 SMALL)
    print("👶 Creating Tiny Student...")
    # <--- CHANGED TO SMALL
    student = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT) 
    
    # Adjust Head
    num_ftrs = student.classifier[3].in_features
    student.classifier[3] = nn.Linear(num_ftrs, num_classes)
    student = student.to(DEVICE)

    # 4. OPTIMIZER
    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. TRAINING LOOP
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        student.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            optimizer.zero_grad()
            student_logits = student(inputs)
            
            loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
            
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f"   Epoch {epoch+1}: Batch {i+1}/{len(train_loader)}...", end='\r')
        
        scheduler.step()

        # Validation
        student.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = student(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        curr_f1 = report['macro avg']['f1-score']
        
        print(f"\n✅ Epoch {epoch+1}/{EPOCHS} | Tiny Student F1: {curr_f1:.2%}")

        if curr_f1 > best_f1:
            best_f1 = curr_f1
            torch.save(student.state_dict(), STUDENT_PATH)
            print(f"   💾 Saved Best Tiny Model! ({STUDENT_PATH})")
            if epoch > 5:
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    elapsed = time.time() - start_time
    print(f"\n🏆 Distillation Complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"📁 Final Tiny Model: {STUDENT_PATH}")

if __name__ == "__main__":
    train_student()