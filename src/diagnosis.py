import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from torchvision import transforms
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod, 
    ProjectedGradientDescent, 
    SquareAttack, 
    AdversarialPatch
)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet Constants (Numpy format for calculations)
MEAN_NP = np.array([0.485, 0.456, 0.406]).astype(np.float32)
STD_NP = np.array([0.229, 0.224, 0.225]).astype(np.float32)

# ImageNet Constants (Tensor format for PyTorch)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class DiagnosisRunner:
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.model.eval()
        
        # Configure ART Classifier
        self.art_classifier = PyTorchClassifier(
            model=self.model,
            clip_values=(0.0, 1.0),
            preprocessing=(MEAN_NP.reshape(3, 1, 1), STD_NP.reshape(3, 1, 1)),
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=0.01),
            input_shape=(3, 224, 224),
            nb_classes=43,
            device_type='gpu' if torch.cuda.is_available() else 'cpu'
        )

    def get_prediction(self, img_tensor):
        """Helper: Get prediction for a tensor [0, 1]."""
        # Ensure tensor is 4D (B, C, H, W)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Apply Normalization strictly for inference
        norm_transform = transforms.Normalize(MEAN, STD)
        input_norm = norm_transform(img_tensor.squeeze(0)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(input_norm)
            
        pred_id = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1).max().item() * 100
        return pred_id, conf

    # ------------------------------------------------------------------
    # PHYSICAL ATTACKS (ALBUMENTATIONS)
    # ------------------------------------------------------------------
    
    def _apply_albumentation(self, x_in, transform):
        # 1. Safety Clip [0, 1]
        x_in = np.clip(x_in, 0.0, 1.0)
        # 2. Convert to Uint8 [0, 255] HWC
        img_hwc = (x_in.transpose((1, 2, 0)) * 255).astype(np.uint8)
        # 3. Transform
        augmented = transform(image=img_hwc)['image']
        # 4. Convert back to Float [0, 1] CHW
        img_chw = augmented.astype(np.float32) / 255.0
        img_chw = img_chw.transpose((2, 0, 1))
        return img_chw

    def _logic_rain(self, x_in, severity):
        transform = A.Compose([
            A.RandomRain(
                brightness_coefficient=1.0 - (0.1 * severity),
                drop_length=10 * severity, drop_width=1, drop_color=(200, 200, 200),
                blur_value=3 + (severity // 2), p=1.0
            )
        ])
        return self._apply_albumentation(x_in, transform)

    def _logic_sun_flare(self, x_in, severity):
        transform = A.Compose([
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                num_flare_circles_lower=1, num_flare_circles_upper=severity + 2,
                src_radius=100 + (50 * severity), src_color=(255, 255, 255), p=1.0
            )
        ])
        return self._apply_albumentation(x_in, transform)

    def _logic_fog(self, x_in, severity):
        transform = A.Compose([
            A.RandomFog(
                fog_coef_lower=0.1 * severity, fog_coef_upper=0.15 * severity, 
                alpha_coef=0.08, p=1.0
            )
        ])
        return self._apply_albumentation(x_in, transform)

    # ------------------------------------------------------------------
    # ART ATTACK WRAPPERS
    # ------------------------------------------------------------------

    def _wrap_patch(self, x_in, severity, target_class):
        scale = 0.15 + (0.05 * severity)
        patch_attack = AdversarialPatch(
            classifier=self.art_classifier, rotation_max=20, scale_min=scale-0.05, scale_max=scale,
            max_iter=50, patch_shape=(3, 60, 60), verbose=False
        )
        y_target = np.zeros((1, 43))
        y_target[0, (target_class + 1) % 43] = 1.0 
        patch, _ = patch_attack.generate(x=np.expand_dims(x_in, 0), y=y_target)
        return patch_attack.apply_patch(np.expand_dims(x_in, 0), patch)[0]

    def _wrap_square(self, x_in, severity, target_class):
        iters = 20 * severity 
        attack = SquareAttack(estimator=self.art_classifier, max_iter=iters, eps=0.1, verbose=False)
        y_true = np.array([target_class])
        return attack.generate(x=np.expand_dims(x_in, 0), y=y_true)[0]


    # ------------------------------------------------------------------
    # MAIN EXECUTION
    # ------------------------------------------------------------------

    def run_diagnosis(self, img_tensor, filename):
        # --- FIX 1: Robust Dimensions Handling ---
        # Ensure we have a numpy array (C, H, W) regardless of input shape
        x_clean_np = img_tensor.cpu().numpy()
        if x_clean_np.ndim == 4:
            x_clean_np = x_clean_np[0] # Remove batch dim if present
        
        # --- FIX 2: Defensive Data Sanitization ---
        # Detect negative values (ImageNet Normalized) -> Denormalize
        if x_clean_np.min() < 0:
            # Formula: x * STD + MEAN
            x_clean_np = x_clean_np.transpose(1, 2, 0) * STD_NP + MEAN_NP
            x_clean_np = x_clean_np.transpose(2, 0, 1)
            
        # Detect 0-255 scale -> Normalize to 0-1
        if x_clean_np.max() > 1.1:
            x_clean_np = x_clean_np / 255.0

        # Clip strictly to [0, 1] to prevent white/black artifacts
        x_clean_np = np.clip(x_clean_np, 0.0, 1.0)
        
        # Re-create sanitized tensor for baseline prediction
        clean_tensor = torch.from_numpy(x_clean_np).float().to(DEVICE)
        y_clean, clean_conf = self.get_prediction(clean_tensor)

        report = {
            "filename": filename,
            "original_class": y_clean,
            "original_conf": clean_conf,
            "attacks": {}
        }

        attacks_map = {
            "FGSM": lambda x, s: FastGradientMethod(self.art_classifier, eps=0.015*s).generate(x=np.expand_dims(x,0))[0],
            "PGD": lambda x, s: ProjectedGradientDescent(self.art_classifier, eps=0.015*s, max_iter=10*s).generate(x=np.expand_dims(x,0))[0],
            "Square": lambda x, s: self._wrap_square(x, s, y_clean),
            "AdvPatch": lambda x, s: self._wrap_patch(x, s, y_clean),
            "Rain": lambda x, s: self._logic_rain(x, s),
            "SunFlare": lambda x, s: self._logic_sun_flare(x, s),
            "Fog": lambda x, s: self._logic_fog(x, s)
        }

        for name, attack_func in attacks_map.items():
            break_severity = 6
            last_generated_adv = x_clean_np.copy()
            
            for severity in range(1, 6):
                try:
                    adv_np = attack_func(x_clean_np, severity)
                    last_generated_adv = adv_np
                    
                    adv_tensor = torch.from_numpy(adv_np).float().to(DEVICE)
                    pred_id, conf = self.get_prediction(adv_tensor)
                    
                    if pred_id != y_clean:
                        break_severity = severity
                        base_score = (severity - 1) * 20
                        bonus_score = max(0, (100 - conf) / 5)
                        final_score = min(base_score + bonus_score, 99.9)

                        report["attacks"][name] = {
                            "status": "Broken",
                            "severity": severity,
                            "final_conf": conf,
                            "adv_image": adv_np,
                            "score": round(final_score, 1)
                        }
                        break 
                except Exception as e:
                    print(f"Error in {name}: {e}")
                    break
            
            if break_severity == 6:
                report["attacks"][name] = {
                    "status": "Robust",
                    "severity": 6,
                    "final_conf": clean_conf,
                    "adv_image": last_generated_adv,
                    "score": 100.0
                }

        return report