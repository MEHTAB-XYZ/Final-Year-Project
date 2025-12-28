import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ================= CONFIG =================
DATA_DIR = r"C:\Users\mehta\OneDrive\Desktop\Final Year Project\auto_robust_xai\data\gtsrb\train"
MODEL_PATH = "models/resnet18_gtsrb_best.pt"

IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 0   # IMPORTANT FOR WINDOWS
# ==========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------- transforms --------
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------- dataset --------
    dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,   # <- critical
        pin_memory=torch.cuda.is_available()
    )

    class_names = dataset.classes
    num_classes = len(class_names)
    print("Number of classes:", num_classes)

    # -------- model --------
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # -------- inference --------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # -------- confusion matrix --------
    cm = confusion_matrix(all_labels, all_preds)

    # -------- plot --------
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        fmt="d",
        cbar=True
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – GTSRB (ResNet18)")
    plt.tight_layout()
    plt.show()

    # -------- per-class accuracy --------
    print("\nPer-class accuracy:\n")
    for i, cls in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0.0
        print(f"{cls:40s} : {acc:.4f}")

    # -------- classification report --------
    print("\nClassification Report:\n")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))


if __name__ == "__main__":
    main()
