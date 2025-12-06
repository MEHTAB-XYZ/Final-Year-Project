# src/utils.py

import torch
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# GTSRB CLASS NAMES (0–42)
# ------------------------------------------------------------
GTSRB_CLASSES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles > 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles > 3.5 tons",
]


# ------------------------------------------------------------
# Convert class index -> full readable label
# ------------------------------------------------------------
def class_name(idx):
    if 0 <= idx < len(GTSRB_CLASSES):
        return GTSRB_CLASSES[idx]
    return f"Class {idx}"


# ------------------------------------------------------------
# Convert PIL → Tensor (Normalization)
# NOTE: Actively used by loader.py
# ------------------------------------------------------------
def pil_to_tensor(pil_img):
    """Convert a PIL image to a normalized tensor."""
    img = np.array(pil_img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean[:, None, None]) / std[:, None, None]
    return torch.tensor(img).float().unsqueeze(0)  # shape: (1,3,H,W)


# ------------------------------------------------------------
# Convert Tensor → PIL image (for digital attacks like FGSM)
# ------------------------------------------------------------
def tensor_to_pil(t):
    """
    Converts an adversarially perturbed tensor back into a displayable PIL image.
    Handles denormalization automatically.
    """
    t = t.squeeze().detach().cpu().numpy()        # (3,H,W)
    t = np.transpose(t, (1, 2, 0))                # HWC

    # Undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    t = t * std + mean

    # Convert to 0–255 range
    t = np.clip(t * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(t)


# ------------------------------------------------------------
# Utility: Convert numpy image → PIL
# ------------------------------------------------------------
def np_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))
