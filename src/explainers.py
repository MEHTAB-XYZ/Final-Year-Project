# src/explainers.py

import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models

def gradcam_visualization(model, img_tensor, layer=None):
    if layer is None:
        layer = model.layer4[-1]  # last conv layer

    cam = GradCAM(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    rgb_img = img_tensor.squeeze().detach().cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1,2,0))
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_img
