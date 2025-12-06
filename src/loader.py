# src/loader.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load Model
# ------------------------------
def load_model(model_path="models/resnet18_gtsrb_state.pt", num_classes=43):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ------------------------------
# Image Preprocessing
# ------------------------------
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_image(img: Image.Image):
    """Convert PIL image → tensor"""
    return img_tf(img).unsqueeze(0).to(DEVICE)


# ------------------------------
# Prediction Function
# ------------------------------
@torch.no_grad()
def predict(model, img_tensor):
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs.max().item()
    return pred, confidence, probs.cpu()
