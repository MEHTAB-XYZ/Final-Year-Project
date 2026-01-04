# src/metrics.py

import numpy as np

def confidence_drop(original_conf, adv_conf):
    return float(original_conf - adv_conf)

def attack_success(orig_label, adv_label):
    return int(orig_label != adv_label)

def robustness_score(original_conf, adv_conf):
    diff = original_conf - adv_conf
    return max(0, 1 - diff)


# ------------------------------
# Grad-CAM Explainability Metrics
# ------------------------------

def cam_drift(cam_clean, cam_adv):
    """
    Mean absolute difference between clean and adversarial CAMs.
    Measures how much the model's attention shifts under attack.
    
    Args:
        cam_clean: Clean image CAM (numpy array)
        cam_adv: Adversarial image CAM (numpy array)
    
    Returns:
        Float: Mean absolute difference
    """
    return float(np.mean(np.abs(cam_clean - cam_adv)))


def cam_focus(cam, threshold=0.6):
    """
    Fraction of pixels with high attention (above threshold).
    Measures attention concentration.
    
    Args:
        cam: CAM heatmap (numpy array)
        threshold: Attention threshold (default 0.6)
    
    Returns:
        Float: Fraction of high-attention pixels [0, 1]
    """
    return float(np.mean(cam >= threshold))


def cam_entropy(cam):
    """
    Entropy of normalized CAM values.
    Measures uncertainty/dispersion of attention distribution.
    
    Args:
        cam: CAM heatmap (numpy array)
    
    Returns:
        Float: Shannon entropy (higher = more dispersed attention)
    """
    eps = 1e-10
    
    # Flatten and normalize to probability distribution
    cam_flat = cam.flatten()
    cam_sum = np.sum(cam_flat)
    
    # Handle edge case of zero CAM
    if cam_sum < eps:
        return 0.0
    
    p = cam_flat / (cam_sum + eps)
    
    # Compute Shannon entropy with numerical stability
    # Filter out zero probabilities to avoid log(0)
    p_nonzero = p[p > eps]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero + eps))
    
    return float(entropy)
