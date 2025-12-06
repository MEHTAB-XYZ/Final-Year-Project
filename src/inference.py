# src/inference.py

from loader import load_model, load_image, predict
from attacks import fgsm_attack, pgd_attack, bim_attack
from physical_attacks import add_fog, add_brightness, add_rain
from explainers import gradcam_visualization
from metrics import confidence_drop, attack_success, robustness_score

def run_inference(model, pil_img, attack="none"):
    img_tensor = load_image(pil_img)

    # Original clean prediction
    orig_label, orig_conf, _ = predict(model, img_tensor)

    # Prepare defaults
    adv_pil = pil_img   # If no attack, attacked image = original

    # -----------------------------------------------------------------
    # DIGITAL ATTACKS
    # -----------------------------------------------------------------
    if attack == "fgsm":
        adv_tensor = fgsm_attack(model, img_tensor, orig_label)
        adv_pil = None

    elif attack == "pgd":
        adv_tensor = pgd_attack(model, img_tensor, orig_label)
        adv_pil = None

    elif attack == "bim":
        adv_tensor = bim_attack(model, img_tensor, orig_label)
        adv_pil = None

    # -----------------------------------------------------------------
    # PHYSICAL ATTACKS (produce PIL images)
    # -----------------------------------------------------------------
    elif attack == "fog":
        adv_pil = add_fog(pil_img)
        adv_tensor = load_image(adv_pil)

    elif attack == "rain":
        adv_pil = add_rain(pil_img)
        adv_tensor = load_image(adv_pil)

    elif attack == "brightness":
        adv_pil = add_brightness(pil_img)
        adv_tensor = load_image(adv_pil)

    else:
        adv_tensor = img_tensor

    # Attack prediction
    adv_label, adv_conf, _ = predict(model, adv_tensor)

    # Metrics
    drop = confidence_drop(orig_conf, adv_conf)
    success = attack_success(orig_label, adv_label)
    robust = robustness_score(orig_conf, adv_conf)

    # GradCAM (always on original)
    cam_img = gradcam_visualization(model, img_tensor)

    return {
        "orig_label": orig_label,
        "orig_conf": orig_conf,
        "adv_label": adv_label,
        "adv_conf": adv_conf,
        "confidence_drop": drop,
        "attack_success": success,
        "robustness_score": robust,
        "gradcam": cam_img,
        "adv_tensor": adv_tensor,
        "adv_pil": adv_pil,
    }
