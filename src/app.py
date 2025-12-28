# src/app.py
'''
import streamlit as st
import os
import torch
import numpy as np
from PIL import Image

# Import existing helpers
# We'll use loader.predict for PyTorch models directly to bypass full inference pipeline
from loader import load_model as load_torch_model, load_image as load_torch_image, predict as predict_torch
from utils import class_name, tensor_to_pil
from attacks import fgsm_attack, pgd_attack, bim_attack
from explainers import gradcam_visualization
from metrics import confidence_drop, attack_success, robustness_score
from physical_attacks import add_fog, add_brightness, add_rain

st.set_page_config(layout="wide")
st.title("AutoRobustXAI – Traffic Sign Classifier")

# ---------------------------------------------------------------------
# 1. MODEL UPLOAD SECTION
# ---------------------------------------------------------------------
st.sidebar.header("Model Settings")
model_file = st.sidebar.file_uploader("Upload Model (.h5 or .pt)", type=["h5", "pt"])

# Global variable to hold the loaded model
model = None
model_type = None

if model_file:
    # Save the uploaded file temporarily
    file_ext = model_file.name.split('.')[-1]
    temp_path = f"temp_uploaded_model.{file_ext}"
    
    with open(temp_path, "wb") as f:
        f.write(model_file.getbuffer())

    try:
        if file_ext == "h5":
            # Lazy import tensorflow/keras to avoid crash if not using it
            import tensorflow as tf
            model = tf.keras.models.load_model(temp_path)
            model_type = "keras"
            st.sidebar.success(f"Keras model loaded: {model_file.name}")
        
        elif file_ext == "pt":
            # Use existing PyTorch loader logic
            # Note: loader.load_model expects a path and loads state_dict
            model = load_torch_model(model_path=temp_path)
            model_type = "torch"
            st.sidebar.success(f"PyTorch model loaded: {model_file.name}")
            
    except Exception as e:
        st.error(f"Error loading model: {e}")

else:
    st.info("Please upload a model file (.h5 or .pt) to begin.")


# ---------------------------------------------------------------------
# 2. IMAGE UPLOAD & CLASSIFICATION
# ---------------------------------------------------------------------
uploaded = st.file_uploader("Upload a Traffic Sign Image", type=["jpg", "png", "jpeg"])

if uploaded and model is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    
    # Display Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(pil_img, width=300)

    # Perform Classification
    with col2:
        st.subheader("Classification Result")
        
        # LOGIC FOR KERAS (.h5)
        if model_type == "keras":
            try:
                # Preprocessing for Keras (assuming ResNet-like 224x224)
                img_array = pil_img.resize((224, 224))
                img_array = np.array(img_array).astype(np.float32) 
                # Normalize? Standard ResNet uses caffe-style or 0-1. 
                # Let's assume standard [0,1] or simple scaling for now. 
                # If the user model is strict, this might precise calibration, but 1/255 is standard.
                img_array = img_array / 255.0  
                img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

                preds = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(preds, axis=1)[0]
                confidence = np.max(preds)
                
                label_str = class_name(pred_idx)
                
                st.markdown(f"**Prediction:** {label_str}")
                st.markdown(f"**Confidence:** {confidence:.4f}")
                
            except Exception as e:
                st.error(f"Inference Error: {e}")

        # LOGIC FOR PYTORCH (.pt)
        elif model_type == "torch":
            try:
                img_tensor = load_torch_image(pil_img)
                pred_idx, confidence, _ = predict_torch(model, img_tensor)
                
                label_str = class_name(pred_idx)
                
                st.markdown(f"**Prediction:** {label_str}")
                st.markdown(f"**Confidence:** {confidence:.4f}")
                
            except Exception as e:
                st.error(f"Inference Error: {e}")


# ---------------------------------------------------------------------
# 3. ADVERSARIAL ATTACKS & GRAD-CAM VISUALIZATION
# ---------------------------------------------------------------------
if uploaded and model is not None:
    st.sidebar.header("Attack Settings")
    
    attack_type = st.sidebar.selectbox(
        "Select Attack Type",
        ["None", "FGSM", "PGD", "BIM", "Fog", "Brightness", "Rain"]
    )
    
    # Attack parameters
    if attack_type == "FGSM":
        eps = st.sidebar.slider("Epsilon (perturbation magnitude)", 0.001, 0.1, 0.01)
    elif attack_type == "PGD":
        eps = st.sidebar.slider("Epsilon", 0.001, 0.1, 0.03)
        alpha = st.sidebar.slider("Alpha (step size)", 0.001, 0.05, 0.01)
        iters = st.sidebar.slider("Iterations", 1, 50, 20)
    elif attack_type == "BIM":
        eps = st.sidebar.slider("Epsilon", 0.001, 0.1, 0.03)
        alpha = st.sidebar.slider("Alpha (step size)", 0.001, 0.05, 0.005)
        iters = st.sidebar.slider("Iterations", 1, 50, 10)
    elif attack_type == "Fog":
        fog_strength = st.sidebar.slider("Fog Strength", 0.0, 1.0, 0.5)
    elif attack_type == "Brightness":
        brightness_val = st.sidebar.slider("Brightness Value", -100, 100, 40)
    
    if attack_type != "None":
        col1, col2 = st.columns(2)
        
        # Original Image & Classification
        with col1:
            st.subheader("Original Image")
            st.image(pil_img, width=300)
            
            if model_type == "torch":
                img_tensor = load_torch_image(pil_img)
                orig_label, orig_conf, _ = predict_torch(model, img_tensor)
                orig_label_str = class_name(orig_label)
                
                st.markdown(f"**Prediction:** {orig_label_str}")
                st.markdown(f"**Confidence:** {orig_conf:.4f}")
        
        # Adversarial Image & Classification
        with col2:
            st.subheader("Adversarial/Attack Image")
            
            if model_type == "torch":
                try:
                    img_tensor = load_torch_image(pil_img)
                    orig_label, orig_conf, _ = predict_torch(model, img_tensor)
                    
                    # Generate adversarial image
                    if attack_type == "FGSM":
                        adv_tensor = fgsm_attack(model, img_tensor, orig_label, eps=eps)
                    elif attack_type == "PGD":
                        adv_tensor = pgd_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                    elif attack_type == "BIM":
                        adv_tensor = bim_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                    elif attack_type == "Fog":
                        adv_pil = add_fog(pil_img, strength=fog_strength)
                        adv_tensor = load_torch_image(adv_pil)
                    elif attack_type == "Brightness":
                        adv_pil = add_brightness(pil_img, value=brightness_val)
                        adv_tensor = load_torch_image(adv_pil)
                    elif attack_type == "Rain":
                        adv_pil = add_rain(pil_img)
                        adv_tensor = load_torch_image(adv_pil)
                    
                    # Convert adversarial tensor to PIL for display
                    if attack_type in ["FGSM", "PGD", "BIM"]:
                        adv_pil = tensor_to_pil(adv_tensor)
                    
                    st.image(adv_pil, width=300)
                    
                    # Adversarial classification
                    adv_label, adv_conf, _ = predict_torch(model, adv_tensor)
                    adv_label_str = class_name(adv_label)
                    
                    st.markdown(f"**Prediction:** {adv_label_str}")
                    st.markdown(f"**Confidence:** {adv_conf:.4f}")
                    
                except Exception as e:
                    st.error(f"Attack Error: {e}")
        
        # Robustness Metrics
        st.subheader("Robustness Metrics")
        
        if model_type == "torch":
            try:
                img_tensor = load_torch_image(pil_img)
                orig_label, orig_conf, _ = predict_torch(model, img_tensor)
                
                # Generate adversarial image again for metrics
                if attack_type == "FGSM":
                    adv_tensor = fgsm_attack(model, img_tensor, orig_label, eps=eps)
                elif attack_type == "PGD":
                    adv_tensor = pgd_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                elif attack_type == "BIM":
                    adv_tensor = bim_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                elif attack_type == "Fog":
                    adv_pil = add_fog(pil_img, strength=fog_strength)
                    adv_tensor = load_torch_image(adv_pil)
                elif attack_type == "Brightness":
                    adv_pil = add_brightness(pil_img, value=brightness_val)
                    adv_tensor = load_torch_image(adv_pil)
                elif attack_type == "Rain":
                    adv_pil = add_rain(pil_img)
                    adv_tensor = load_torch_image(adv_pil)
                
                adv_label, adv_conf, _ = predict_torch(model, adv_tensor)
                
                # Calculate metrics
                conf_drop = confidence_drop(orig_conf, adv_conf)
                attack_succ = attack_success(orig_label, adv_label)
                robust_score = robustness_score(orig_conf, adv_conf)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric("Confidence Drop", f"{conf_drop:.4f}")
                
                with col_m2:
                    st.metric("Attack Success", f"{'Yes' if attack_succ else 'No'}")
                
                with col_m3:
                    st.metric("Robustness Score", f"{robust_score:.4f}")
                    
            except Exception as e:
                st.error(f"Metrics Error: {e}")


# ---------------------------------------------------------------------
# 4. GRAD-CAM VISUALIZATION
# ---------------------------------------------------------------------
if uploaded and model is not None and model_type == "torch":
    st.sidebar.header("Explainability Settings")
    
    if st.sidebar.checkbox("Show Grad-CAM Visualization"):
        st.subheader("Grad-CAM Heatmap")
        
        try:
            img_tensor = load_torch_image(pil_img)
            cam_img = gradcam_visualization(model, img_tensor)
            st.image(cam_img, use_column_width=True)
            
            st.info("Grad-CAM highlights the regions in the image that contributed most to the model's prediction.")
            
        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")


'''
# src/app.py
import streamlit as st
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

# --- IMPORTS FROM YOUR OLD APP ---
from loader import load_model as load_torch_model, load_image as load_torch_image, predict as predict_torch
from utils import class_name, tensor_to_pil
from attacks import fgsm_attack, pgd_attack, bim_attack
from explainers import gradcam_visualization
from metrics import confidence_drop, attack_success, robustness_score
from physical_attacks import add_fog, add_brightness, add_rain

# --- NEW IMPORTS FOR DIAGNOSIS ---
from diagnosis import DiagnosisRunner
import torchvision.transforms as transforms

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AutoRobustXAI")
st.title("AutoRobustXAI – Traffic Sign Defense System")

# ---------------------------------------------------------------------
# 1. GLOBAL MODEL UPLOAD (Sidebar) - Shared by both tabs
# ---------------------------------------------------------------------
st.sidebar.header("Model Settings")
model_file = st.sidebar.file_uploader("Upload Model (.h5 or .pt)", type=["h5", "pt"])

# Global variable to hold the loaded model
model = None
model_type = None

if model_file:
    # Save the uploaded file temporarily
    file_ext = model_file.name.split('.')[-1]
    temp_path = f"temp_uploaded_model.{file_ext}"
    
    with open(temp_path, "wb") as f:
        f.write(model_file.getbuffer())

    try:
        if file_ext == "h5":
            # Lazy import tensorflow/keras
            import tensorflow as tf
            model = tf.keras.models.load_model(temp_path)
            model_type = "keras"
            st.sidebar.success(f"Keras model loaded: {model_file.name}")
        
        elif file_ext == "pt":
            # Use existing PyTorch loader logic
            model = load_torch_model(model_path=temp_path)
            model_type = "torch"
            st.sidebar.success(f"PyTorch model loaded: {model_file.name}")
            
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

else:
    st.sidebar.info("Please upload a model file (.h5 or .pt) to begin.")


# ---------------------------------------------------------------------
# 2. MAIN INTERFACE TABS
# ---------------------------------------------------------------------
tab_sandbox, tab_diagnosis = st.tabs(["🧪 Sandbox Mode (Manual)", "🛡️ Diagnosis Mode (Automated)"])


# =====================================================================
# TAB 1: SANDBOX MODE (EXACT COPY OF YOUR OLD APP LOGIC)
# =====================================================================
with tab_sandbox:
    st.header("Interactive Robustness Testing")
    # ---------------------------------------------------------------------
    # IMAGE UPLOAD & CLASSIFICATION
    # ---------------------------------------------------------------------
    uploaded = st.file_uploader("Upload Single Image", type=["jpg", "png", "jpeg"], key="sandbox_up")

    if uploaded and model is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        
        # Display Image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(pil_img, width=300)

        # Perform Classification
        with col2:
            st.subheader("Classification Result")
            
            # LOGIC FOR KERAS (.h5)
            if model_type == "keras":
                try:
                    img_array = pil_img.resize((224, 224))
                    img_array = np.array(img_array).astype(np.float32) 
                    img_array = img_array / 255.0  
                    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

                    preds = model.predict(img_array, verbose=0)
                    pred_idx = np.argmax(preds, axis=1)[0]
                    confidence = np.max(preds)
                    
                    label_str = class_name(pred_idx)
                    
                    st.markdown(f"**Prediction:** {label_str}")
                    st.markdown(f"**Confidence:** {confidence:.4f}")
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")

            # LOGIC FOR PYTORCH (.pt)
            elif model_type == "torch":
                try:
                    img_tensor = load_torch_image(pil_img)
                    pred_idx, confidence, _ = predict_torch(model, img_tensor)
                    
                    label_str = class_name(pred_idx)
                    
                    st.markdown(f"**Prediction:** {label_str}")
                    st.markdown(f"**Confidence:** {confidence:.4f}")
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")


        # ---------------------------------------------------------------------
        # ADVERSARIAL ATTACKS & GRAD-CAM VISUALIZATION
        # ---------------------------------------------------------------------
        st.divider()
        st.subheader("Apply Attack")
        
        attack_type = st.selectbox(
            "Select Attack Type",
            ["None", "FGSM", "PGD", "BIM", "Fog", "Brightness", "Rain"],
            key="sb_attack_select"
        )
        
        # Attack parameters
        eps = 0.01
        alpha = 0.01
        iters = 10
        fog_strength = 0.5
        brightness_val = 40

        if attack_type == "FGSM":
            eps = st.slider("Epsilon (perturbation magnitude)", 0.001, 0.1, 0.01)
        elif attack_type == "PGD":
            eps = st.slider("Epsilon", 0.001, 0.1, 0.03)
            alpha = st.slider("Alpha (step size)", 0.001, 0.05, 0.01)
            iters = st.slider("Iterations", 1, 50, 20)
        elif attack_type == "BIM":
            eps = st.slider("Epsilon", 0.001, 0.1, 0.03)
            alpha = st.slider("Alpha (step size)", 0.001, 0.05, 0.005)
            iters = st.slider("Iterations", 1, 50, 10)
        elif attack_type == "Fog":
            fog_strength = st.slider("Fog Strength", 0.0, 1.0, 0.5)
        elif attack_type == "Brightness":
            brightness_val = st.slider("Brightness Value", -100, 100, 40)
        
        if attack_type != "None":
            col1, col2 = st.columns(2)
            
            # Original Image & Classification (Recalculated for layout consistency)
            with col1:
                st.subheader("Original Image")
                st.image(pil_img, width=300)
                
                if model_type == "torch":
                    img_tensor = load_torch_image(pil_img)
                    orig_label, orig_conf, _ = predict_torch(model, img_tensor)
                    orig_label_str = class_name(orig_label)
                    
                    st.markdown(f"**Prediction:** {orig_label_str}")
                    st.markdown(f"**Confidence:** {orig_conf:.4f}")
            
            # Adversarial Image & Classification
            with col2:
                st.subheader("Adversarial/Attack Image")
                
                if model_type == "torch":
                    try:
                        img_tensor = load_torch_image(pil_img)
                        orig_label, orig_conf, _ = predict_torch(model, img_tensor)
                        
                        # Generate adversarial image
                        adv_tensor = img_tensor # Default fallback
                        
                        if attack_type == "FGSM":
                            adv_tensor = fgsm_attack(model, img_tensor, orig_label, eps=eps)
                        elif attack_type == "PGD":
                            adv_tensor = pgd_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                        elif attack_type == "BIM":
                            adv_tensor = bim_attack(model, img_tensor, orig_label, eps=eps, alpha=alpha, iters=iters)
                        elif attack_type == "Fog":
                            adv_pil = add_fog(pil_img, strength=fog_strength)
                            adv_tensor = load_torch_image(adv_pil)
                        elif attack_type == "Brightness":
                            adv_pil = add_brightness(pil_img, value=brightness_val)
                            adv_tensor = load_torch_image(adv_pil)
                        elif attack_type == "Rain":
                            adv_pil = add_rain(pil_img)
                            adv_tensor = load_torch_image(adv_pil)
                        
                        # Convert adversarial tensor to PIL for display
                        if attack_type in ["FGSM", "PGD", "BIM"]:
                            adv_pil = tensor_to_pil(adv_tensor)
                        
                        st.image(adv_pil, width=300)
                        
                        # Adversarial classification
                        adv_label, adv_conf, _ = predict_torch(model, adv_tensor)
                        adv_label_str = class_name(adv_label)
                        
                        st.markdown(f"**Prediction:** {adv_label_str}")
                        st.markdown(f"**Confidence:** {adv_conf:.4f}")
                        
                    except Exception as e:
                        st.error(f"Attack Error: {e}")
            
            # Robustness Metrics
            st.subheader("Robustness Metrics")
            
            if model_type == "torch":
                try:
                    # Calculate metrics based on the variables computed above
                    conf_drop = confidence_drop(orig_conf, adv_conf)
                    attack_succ = attack_success(orig_label, adv_label)
                    robust_score = robustness_score(orig_conf, adv_conf)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("Confidence Drop", f"{conf_drop:.4f}")
                    
                    with col_m2:
                        st.metric("Attack Success", f"{'Yes' if attack_succ else 'No'}")
                    
                    with col_m3:
                        st.metric("Robustness Score", f"{robust_score:.4f}")
                        
                except Exception as e:
                    st.error(f"Metrics Error: {e}")

        # Grad-CAM
        if model_type == "torch":
            st.sidebar.header("Explainability Settings")
            if st.sidebar.checkbox("Show Grad-CAM Visualization"):
                st.subheader("Grad-CAM Heatmap")
                try:
                    img_tensor = load_torch_image(pil_img)
                    cam_img = gradcam_visualization(model, img_tensor)
                    st.image(cam_img, use_column_width=True)
                    st.info("Grad-CAM highlights the regions in the image that contributed most to the model's prediction.")
                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")


# =====================================================================
# TAB 2: DIAGNOSIS MODE (Automated)
# =====================================================================
with tab_diagnosis:
    st.header("🛡️ Automated Robustness Diagnosis")
    st.markdown("""
    **Batch Stress Testing:** Upload multiple images. The system will run **7 different attacks** (Gradient, Patch, Physical) 
    at increasing severity (Level 1-5).
    """)
    
    st.info("💡 **Tip:** For the most accurate diagnosis, upload **10-20 images** containing a mix of different signs.")
    
    if model_type != "torch":
        st.warning("⚠️ Diagnosis Mode currently requires a PyTorch (.pt) model.")
    else:
        diag_files = st.file_uploader("Upload Test Images (Batch)", type=["jpg", "png"], accept_multiple_files=True, key="diag_up")
        
        if st.button("🚀 Run Diagnosis") and diag_files:
            runner = DiagnosisRunner(model)
            
            prog_bar = st.progress(0)
            status_txt = st.empty()
            
            all_reports = []
            
            # --- FIX: RAW TRANSFORM ---
            # We strictly want [0, 1] range. 
            # We do NOT use 'load_torch_image' here because it likely normalizes (subtracts mean), 
            # causing the Double Normalization bug.
            raw_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor() # Converts [0, 255] -> [0.0, 1.0]
            ])
            
            for i, file in enumerate(diag_files):
                status_txt.text(f"Diagnosing {file.name}...")
                try:
                    img_pil = Image.open(file).convert('RGB')
                    # Manual Batch Dimension: (1, 3, 224, 224)
                    # This prevents the "squeeze" error
                    img_tensor = raw_transform(img_pil).unsqueeze(0) 
                    
                    report = runner.run_diagnosis(img_tensor, file.name)
                    all_reports.append(report)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                
                prog_bar.progress((i + 1) / len(diag_files))
            
            status_txt.text("Diagnosis Complete!")
            
            if not all_reports:
                st.stop()

            # --- REPORTING ---
            st.divider()
            
            total_score = 0
            total_attacks_count = 0 
            attack_performance = {}
            
            for r in all_reports:
                for atk_name, data in r["attacks"].items():
                    total_score += data["score"]
                    total_attacks_count += 1
                    if atk_name not in attack_performance:
                        attack_performance[atk_name] = []
                    attack_performance[atk_name].append(data["score"])
            
            # Robust Average Calculation
            avg_score = total_score / max(1, total_attacks_count)
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Overall Robustness", f"{avg_score:.1f}/100")
            
            avg_atk_scores = {k: sum(v)/len(v) for k, v in attack_performance.items()}
            weakest = min(avg_atk_scores, key=avg_atk_scores.get)
            strongest = max(avg_atk_scores, key=avg_atk_scores.get)
            
            kpi2.metric("Most Vulnerable To", weakest, f"{avg_atk_scores[weakest]:.1f}")
            kpi3.metric("Strongest Against", strongest, f"{avg_atk_scores[strongest]:.1f}")
            
            st.subheader("Vulnerability Profile")
            chart_data = pd.DataFrame.from_dict(avg_atk_scores, orient='index', columns=['Robustness Score'])
            st.bar_chart(chart_data)
            
            st.subheader("📸 Failure Analysis")
            for r in all_reports:
                with st.expander(f"Report: {r['filename']} (Orig: {class_name(r['original_class'])})"):
                    
                    # Robust Grid Layout (Modulo 4)
                    cols = st.columns(4)
                    
                    for idx, (atk_name, data) in enumerate(r['attacks'].items()):
                        with cols[idx % 4]:
                            disp_img = data['adv_image'].transpose(1, 2, 0)
                            
                            # Safety Clip [0, 1] (Fixes white/black artifacts)
                            disp_img = np.clip(disp_img, 0.0, 1.0)
                            
                            st.image(disp_img, caption=f"{atk_name}\nScore: {data['score']}")
                            
                            if data['status'] == "Broken":
                                st.caption(f"❌ Fail Lvl {data['severity']}")
                            else:
                                st.caption("✅ Robust")