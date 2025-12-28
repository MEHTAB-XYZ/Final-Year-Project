import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ================= CONFIG =================
DATA_DIR = r"C:\Users\mehta\OneDrive\Desktop\Final Year Project\auto_robust_xai\data\gtsrb\train"
SAVE_PATH = "models/resnet18_gtsrb_best.pt"

IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
USE_MIXED_PRECISION = True
SEED = 42
# ==========================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= UTILS ==================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ================= TRANSFORMS =================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ================= DATA ==================
def prepare_dataloaders(data_dir, train_tf, val_tf):
    full_dataset = datasets.ImageFolder(data_dir)
    targets = np.array(full_dataset.targets)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    train_ds = Subset(
        datasets.ImageFolder(data_dir, transform=train_tf),
        train_idx
    )
    val_ds = Subset(
        datasets.ImageFolder(data_dir, transform=val_tf),
        val_idx
    )

    num_workers = min(8, os.cpu_count())
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, full_dataset.classes, targets[train_idx]

# ================= MODEL ==================
def build_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ================= TRAIN / VAL ==================
def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_acc, n = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, labels) * bs
        n += bs

    return total_loss / n, total_acc / n

# ================= MAIN ==================
def main():
    device = get_device()
    print("Device:", device)

    train_tf, val_tf = get_transforms()
    train_loader, val_loader, classes, train_targets = prepare_dataloaders(
        DATA_DIR, train_tf, val_tf
    )

    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_targets),
        y=train_targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and device == "cuda" else None

    best_acc = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Best model saved")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("⏹ Early stopping")
            break

    print(f"\nTraining complete | Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
