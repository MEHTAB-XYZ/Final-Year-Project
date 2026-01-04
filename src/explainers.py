# src/explainers.py

import torch
import torch.nn.functional as F
import numpy as np


# ------------------------------
# Grad-CAM Implementation
# ------------------------------

class GradCAM:
    """
    Self-contained Grad-CAM implementation for PyTorch models.
    
    Compatible with ResNet-style architectures. Returns raw CAM as 
    normalized NumPy arrays for use in metrics and analysis.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model (must be in eval mode)
            target_layer: Target convolutional layer (e.g., model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input image tensor (shape: [1, C, H, W])
            target_class: Target class index (None = use predicted class)
        
        Returns:
            NumPy array: CAM normalized to [0, 1], same spatial size as input
        """
        # Ensure input requires gradients
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Select target class (predicted class if not specified)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients and backpropagate
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        # Global average pooling of gradients (importance weights)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1)
        
        # Apply ReLU (only keep positive contributions)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        
        # Convert to NumPy and normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# ------------------------------
# Helper Function for Visualization
# ------------------------------

def overlay_cam_on_image(image_np, cam, alpha=0.5):
    """
    Overlay CAM heatmap on image (optional visualization helper).
    
    Args:
        image_np: Original image as NumPy array (H, W, 3), RGB, [0, 255] or [0, 1]
        cam: CAM heatmap as NumPy array (H, W), normalized [0, 1]
        alpha: Blending weight for overlay (default 0.5)
    
    Returns:
        NumPy array: Blended image with heatmap overlay (H, W, 3), uint8
    """
    import cv2
    
    # Normalize image to [0, 255] if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Resize CAM to match image dimensions
    cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    
    # Convert CAM to heatmap (JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend heatmap with original image
    overlay = alpha * heatmap + (1 - alpha) * image_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


# ------------------------------
# Legacy Compatibility Wrapper
# ------------------------------

def gradcam_visualization(model, img_tensor, layer=None):
    """
    Legacy wrapper for backward compatibility with existing app.py code.
    
    Args:
        model: PyTorch model
        img_tensor: Input tensor
        layer: Target layer (defaults to model.layer4[-1])
    
    Returns:
        RGB image with CAM overlay (for display in Streamlit)
    """
    if layer is None:
        layer = model.layer4[-1]
    
    # Generate CAM
    gradcam = GradCAM(model, layer)
    cam = gradcam.generate(img_tensor)
    
    # Convert input tensor to image for overlay
    rgb_img = img_tensor.squeeze().detach().cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    # Overlay and return
    return overlay_cam_on_image(rgb_img, cam)