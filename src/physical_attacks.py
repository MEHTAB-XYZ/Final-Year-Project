# src/physical_attacks.py

import cv2
import numpy as np
from PIL import Image

def add_fog(img, strength=0.5):
    img = np.array(img).astype(np.float32)
    fog = np.full_like(img, 255)
    out = cv2.addWeighted(img, 1-strength, fog, strength, 0)
    return Image.fromarray(out.astype(np.uint8))

def add_brightness(img, value=40):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    hsv[...,2] = np.clip(hsv[...,2] + value, 0, 255)
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

def add_rain(img):
    img_np = np.array(img)
    h,w,_ = img_np.shape
    rain = np.random.randint(150,255,(h,w))
    rain = cv2.GaussianBlur(rain,(7,7),0)
    rain = rain[...,None].repeat(3,axis=2)
    out = cv2.addWeighted(img_np, 1, rain.astype(np.uint8), 0.2, 0)
    return Image.fromarray(out)
