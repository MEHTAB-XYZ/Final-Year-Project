# train_model.py — optimized for GPU, full GTSRB, mixed precision, early stopping
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =========== CONFIG ==============
DATA_DIR = "data/gtsrb/train"                 # relative to auto_robust_xai/
SAVE_PATH = "models/resnet18_gtsrb_state.pt"
IMG_SIZE = 224
BATCH_SIZE = 128         # try 128, reduce if OOM
EPOCHS = 20              # you can lower for quick experiments
LR = 1e-3
PATIENCE = 4             # early stopping patience on val acc
WEIGHT_DECAY = 1e-4
USE_MIXED_PRECISION = True
# ==================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_transforms(img_size=IMG_SIZE):
    # Minimal augmentations to keep dataloader fast; you can re-enable more later.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def prepare_dataloaders(data_dir, train_tf, val_tf, batch_size):
    full = datasets.ImageFolder(data_dir, transform=train_tf)
    n = len(full)
    if n == 0:
        raise RuntimeError(f"No images found in {data_dir} — check path.")
    # train/val split (90/10)
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    # ensure val uses val_tf
    val_ds.dataset.transform = val_tf

    num_workers = min(8, max(0, os.cpu_count() - 1))  # sensible default
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, full.classes

def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def train_one_epoch(model, loader, opt, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        opt.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, labels) * bs
        n_samples += bs
        pbar.set_postfix(loss=running_loss / n_samples, acc=running_acc / n_samples)
    return running_loss / n_samples, running_acc / n_samples

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Val  ", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, labels) * bs
        n_samples += bs
        pbar.set_postfix(loss=running_loss / n_samples, acc=running_acc / n_samples)
    return running_loss / n_samples, running_acc / n_samples

def save_state(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    device = get_device()
    print("Device:", device)
    torch.backends.cudnn.benchmark = True  # speed boost for fixed-size inputs

    train_tf, val_tf = make_transforms(IMG_SIZE)
    train_loader, val_loader, classes = prepare_dataloaders(DATA_DIR, train_tf, val_tf, BATCH_SIZE)
    num_classes = len(classes)
    print(f"Found {len(train_loader.dataset) + len(val_loader.dataset)} images, {num_classes} classes")
    print("Classes (first 10):", classes[:10])

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler() if (USE_MIXED_PRECISION and device == "cuda") else None
    print("Mixed precision:", bool(scaler))

    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # reporting
        epoch_time = time.time() - t0
        total_elapsed = time.time() - start_time
        print(f"Epoch {epoch} summary: Train loss {train_loss:.4f}, Train acc {train_acc:.4f}")
        print(f"                 Val   loss {val_loss:.4f}, Val   acc {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s | Total elapsed: {total_elapsed/60:.2f} min")

        # scheduler step (if using ReduceLROnPlateau)
        scheduler.step(val_acc)

        # save best model
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_state(model, SAVE_PATH)
            print(f"New best val acc: {best_val_acc:.4f} — model saved to {SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (patience {PATIENCE})")

        # early stopping
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"\nTraining finished. Best val acc: {best_val_acc:.4f}")
    print("Final model path:", SAVE_PATH)

if __name__ == "__main__":
    main()
