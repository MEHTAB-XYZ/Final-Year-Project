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
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Import Grad-CAM metrics and explainability
from metrics import cam_drift, cam_focus, cam_entropy
from explainers import GradCAM

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet Constants (Numpy format for calculations)
MEAN_NP = np.array([0.485, 0.456, 0.406]).astype(np.float32)
STD_NP = np.array([0.229, 0.224, 0.225]).astype(np.float32)

# ImageNet Constants (Tensor format for PyTorch)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ------------------------------
# Retraining Recommendation System
# ------------------------------

def recommend_retraining(failure_types):
    """
    Generate actionable retraining recommendations based on detected failure types.
    
    Args:
        failure_types: List of detected failure types from Grad-CAM analysis
    
    Returns:
        List of human-readable retraining recommendations
    """
    recommendations = []
    
    if "Attention Drift" in failure_types:
        recommendations.append("Use PGD-based adversarial training to stabilize attention")
    
    if "Patch Dominance" in failure_types:
        recommendations.append("Apply CutMix or Random Erasing to reduce local shortcut learning")
    
    if "Diffuse Attention" in failure_types:
        recommendations.append("Increase training epochs or input resolution to improve feature learning")
    
    if "No Critical Failure" in failure_types:
        recommendations.append("Model attention appears stable; no immediate retraining required")
    
    return recommendations


# ------------------------------
# PDF Report Generation
# ------------------------------

def generate_pdf_report(diagnosis_results, output_path="reports/model_health_report.pdf"):
    """
    Generate a comprehensive PDF health report from batch diagnosis results.
    
    Args:
        diagnosis_results: List of diagnosis report dictionaries
        output_path: Path to save the PDF report
    """
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("AutoRobustXAI – Model Health Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))
    
    # --- SUMMARY STATISTICS ---
    story.append(Paragraph("Summary Statistics", heading_style))
    
    # Aggregate metrics
    total_attacks = 0
    total_conf_drop = 0
    total_robustness = 0
    total_cam_drift = 0
    cam_drift_count = 0
    
    for report in diagnosis_results:
        for attack_name, attack_data in report.get("attacks", {}).items():
            total_attacks += 1
            
            # Calculate confidence drop
            orig_conf = report.get("original_conf", 0)
            final_conf = attack_data.get("final_conf", 0)
            total_conf_drop += (orig_conf - final_conf)
            
            # Robustness score
            total_robustness += attack_data.get("score", 0)
            
            # CAM drift (only if available)
            drift = attack_data.get("cam_drift")
            if drift is not None:
                total_cam_drift += drift
                cam_drift_count += 1
    
    # Compute averages
    avg_conf_drop = total_conf_drop / max(1, total_attacks)
    avg_robustness = total_robustness / max(1, total_attacks)
    avg_cam_drift = total_cam_drift / max(1, cam_drift_count) if cam_drift_count > 0 else 0
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Tests", str(total_attacks)],
        ["Avg Confidence Drop", f"{avg_conf_drop:.2f}%"],
        ["Avg Robustness Score", f"{avg_robustness:.2f}"],
        ["Avg CAM Drift", f"{avg_cam_drift:.4f}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * inch))
    
    # --- FAILURE ANALYSIS ---
    story.append(Paragraph("Failure Analysis", heading_style))
    
    # Count failure types
    failure_counts = {}
    for report in diagnosis_results:
        for attack_name, attack_data in report.get("attacks", {}).items():
            for failure_type in attack_data.get("failure_types", []):
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    
    if failure_counts:
        failure_data = [["Failure Type", "Count"]]
        for failure_type, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
            failure_data.append([failure_type, str(count)])
        
        failure_table = Table(failure_data, colWidths=[3*inch, 2*inch])
        failure_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(failure_table)
    else:
        story.append(Paragraph("No failures detected.", styles['Normal']))
    
    story.append(Spacer(1, 0.4 * inch))
    
    # --- RETRAINING RECOMMENDATIONS ---
    story.append(Paragraph("Retraining Recommendations", heading_style))
    
    # Aggregate unique recommendations
    all_recommendations = set()
    for report in diagnosis_results:
        for attack_name, attack_data in report.get("attacks", {}).items():
            for rec in attack_data.get("recommendations", []):
                all_recommendations.add(rec)
    
    if all_recommendations:
        for i, rec in enumerate(sorted(all_recommendations), 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("No recommendations available.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report generated: {output_path}")


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
                        
                        # --- Grad-CAM Analysis for Failure Detection ---
                        failure_types = []
                        drift = None
                        focus = None
                        entropy = None
                        
                        try:
                            # Initialize Grad-CAM (target last conv layer)
                            gradcam = GradCAM(self.model, self.model.layer4[-1])
                            
                            # Prepare normalized tensors for Grad-CAM
                            norm_transform = transforms.Normalize(MEAN, STD)
                            
                            # Clean CAM
                            clean_tensor_norm = norm_transform(clean_tensor.squeeze(0)).unsqueeze(0)
                            clean_tensor_norm.requires_grad_()
                            cam_clean = gradcam.generate(clean_tensor_norm)
                            
                            # Adversarial CAM
                            adv_tensor_norm = norm_transform(adv_tensor.squeeze(0)).unsqueeze(0)
                            adv_tensor_norm.requires_grad_()
                            cam_adv = gradcam.generate(adv_tensor_norm)
                            
                            # Compute CAM metrics
                            drift = cam_drift(cam_clean, cam_adv)
                            focus = cam_focus(cam_adv)
                            entropy = cam_entropy(cam_adv)
                            
                            # Detect failure types using heuristic thresholds
                            if drift > 0.25:
                                failure_types.append("Attention Drift")
                            if focus > 0.15:
                                failure_types.append("Patch Dominance")
                            if entropy > 4.0:
                                failure_types.append("Diffuse Attention")
                            
                            if not failure_types:
                                failure_types.append("No Critical Failure")
                                
                        except Exception as cam_error:
                            print(f"Grad-CAM analysis error for {name}: {cam_error}")
                            failure_types = ["Analysis Failed"]
                        
                        # Generate retraining recommendations
                        recommendations = recommend_retraining(failure_types)
                        
                        # Create comprehensive diagnosis result
                        diagnosis_result = {
                            "status": "Broken",
                            "severity": severity,
                            "final_conf": conf,
                            "adv_image": adv_np,
                            "score": round(final_score, 1),
                            "failure_types": failure_types,
                            "recommendations": recommendations,
                            "cam_drift": drift,
                            "cam_focus": focus,
                            "cam_entropy": entropy
                        }

                        report["attacks"][name] = diagnosis_result
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