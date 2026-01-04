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
from diagnosis import DiagnosisRunner, generate_pdf_report
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
        
        # Initialize session state for diagnosis results
        if 'all_reports' not in st.session_state:
            st.session_state.all_reports = None
        
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
                    # Store clean tensor for later Grad-CAM analysis
                    report['clean_tensor'] = img_tensor
                    all_reports.append(report)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                
                prog_bar.progress((i + 1) / len(diag_files))
            
            status_txt.text("Diagnosis Complete!")
            
            # Store results in session state
            st.session_state.all_reports = all_reports
            
            if not all_reports:
                st.stop()

        # --- REPORTING SECTION (uses session state) ---
        # Display results if diagnosis has been run
        if st.session_state.all_reports is not None and len(st.session_state.all_reports) > 0:
            all_reports = st.session_state.all_reports
            
            # ============================================================
            # SECTION 1: MODEL ROBUSTNESS SUMMARY
            # ============================================================
            st.divider()
            st.subheader("📊 Model Robustness Summary")
            st.caption("Aggregate performance metrics across all tested attacks")
            
            # Calculate metrics
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
            
            avg_score = total_score / max(1, total_attacks_count)
            avg_atk_scores = {k: sum(v)/len(v) for k, v in attack_performance.items()}
            weakest = min(avg_atk_scores, key=avg_atk_scores.get)
            strongest = max(avg_atk_scores, key=avg_atk_scores.get)
            
            # Display KPIs with captions
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Overall Robustness", f"{avg_score:.1f}/100")
                st.caption("Average robustness across all attacks")
            with kpi2:
                st.metric("Most Vulnerable To", weakest, f"{avg_atk_scores[weakest]:.1f}")
                st.caption("Lowest-scoring attack category")
            with kpi3:
                st.metric("Strongest Against", strongest, f"{avg_atk_scores[strongest]:.1f}")
                st.caption("Highest-performing attack category")
            
            # ============================================================
            # SECTION 2: ATTACK VULNERABILITY PROFILE
            # ============================================================
            st.divider()
            st.subheader("🎯 Attack Vulnerability Profile")
            st.caption("Comparative robustness scores by attack type (higher is better)")
            
            # Sort attacks by score for better visualization
            sorted_attacks = sorted(avg_atk_scores.items(), key=lambda x: x[1], reverse=True)
            attack_names = [name for name, _ in sorted_attacks]
            scores = [score for _, score in sorted_attacks]
            
            # Create color gradient (red for low scores, yellow for medium, green for high)
            colors = []
            for score in scores:
                if score >= 80:
                    colors.append('#2ecc71')  # Green
                elif score >= 60:
                    colors.append('#f39c12')  # Orange
                elif score >= 40:
                    colors.append('#e67e22')  # Dark Orange
                else:
                    colors.append('#e74c3c')  # Red
            
            # Create Plotly bar chart
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=attack_names,
                    y=scores,
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(0,0,0,0.3)', width=1)
                    ),
                    text=[f'{s:.1f}' for s in scores],
                    textposition='outside',
                    textfont=dict(size=12, color='#2c3e50'),
                    hovertemplate='<b>%{x}</b><br>Robustness Score: %{y:.1f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                xaxis_title="Attack Type",
                yaxis_title="Robustness Score",
                yaxis=dict(range=[0, 105], gridcolor='rgba(0,0,0,0.1)'),
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=50, r=50, t=30, b=100),
                font=dict(family="Arial, sans-serif", size=12, color='#2c3e50'),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=13,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")  # Spacing
            
            # ============================================================
            # SECTION 3: EXPORT & CONFIGURATION
            # ============================================================
            st.divider()
            
            # PDF Report Generation
            col_pdf, col_xai = st.columns([2, 1])
            
            with col_pdf:
                if st.button("📄 Generate & Download Model Health Report (PDF)", use_container_width=True):
                    try:
                        generate_pdf_report(all_reports)
                        st.success("Model Health Report generated successfully.")
                        
                        pdf_path = "reports/model_health_report.pdf"
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_file,
                                    file_name="model_health_report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")
            
            with col_xai:
                show_xai = st.checkbox("Show Explainability Analysis", value=False)
                st.caption("Enable Grad-CAM visualizations and retraining suggestions")
            
            # ============================================================
            # SECTION 4: PER-IMAGE DIAGNOSIS & EXPLAINABILITY
            # ============================================================
            st.divider()
            st.subheader("� Per-Image Diagnosis & Explainability")
            st.caption(f"Detailed analysis for {len(all_reports)} tested images")
            
            st.write("")  # Spacing
            
            for r in all_reports:
                with st.expander(f"**{r['filename']}** — Original: {class_name(r['original_class'])}", expanded=False):
                    
                    # Attack Results Grid
                    st.markdown("**Attack Test Results**")
                    cols = st.columns(4)
                    
                    for idx, (atk_name, data) in enumerate(r['attacks'].items()):
                        with cols[idx % 4]:
                            disp_img = data['adv_image'].transpose(1, 2, 0)
                            disp_img = np.clip(disp_img, 0.0, 1.0)
                            
                            st.image(disp_img, caption=f"{atk_name}\nScore: {data['score']}")
                            
                            if data['status'] == "Broken":
                                st.caption(f"❌ Failed at Level {data['severity']}")
                            else:
                                st.caption("✅ Robust")
                    
                    st.write("")  # Spacing
                    
                    # ========================================================
                    # EXPLAINABILITY SECTION (Only if enabled)
                    # ========================================================
                    if show_xai:
                        st.divider()
                        st.markdown("### Explainability Analysis")
                        
                        # Attack Selection
                        available_attacks = list(r['attacks'].keys())
                        selected_attack = st.selectbox(
                            "Select attack to analyze:",
                            available_attacks,
                            key=f"attack_select_{r['filename']}"
                        )
                        
                        if selected_attack:
                            atk_data = r['attacks'][selected_attack]
                            
                            # Grad-CAM Visualization
                            st.markdown("**Model Attention Comparison**")
                            
                            try:
                                device = next(model.parameters()).device
                                clean_tensor = r.get('clean_tensor')
                                
                                if clean_tensor is not None:
                                    clean_tensor = clean_tensor.to(device)
                                
                                adv_tensor = torch.tensor(atk_data['adv_image']).unsqueeze(0).float().to(device)
                                
                                if clean_tensor is not None:
                                    cam_clean = gradcam_visualization(model, clean_tensor)
                                    cam_adv = gradcam_visualization(model, adv_tensor)
                                    
                                    # Side-by-side Grad-CAM
                                    col_cam1, col_cam2 = st.columns(2)
                                    with col_cam1:
                                        st.image(cam_clean, use_column_width=True)
                                        st.caption("Model Attention (Clean Input)")
                                    with col_cam2:
                                        st.image(cam_adv, use_column_width=True)
                                        st.caption(f"Model Attention (After {selected_attack})")
                                    
                                    st.write("")  # Spacing
                                    
                                    # Attack Status
                                    if atk_data['status'] == "Broken":
                                        st.error(f"Attack **{selected_attack}** broke the model at severity level {atk_data['severity']}")
                                    else:
                                        st.success(f"Model remained robust against **{selected_attack}**")
                                    
                                    # CAM Metrics
                                    if 'cam_drift' in atk_data and 'cam_focus' in atk_data and 'cam_entropy' in atk_data:
                                        st.markdown("**Attention Metrics**")
                                        met_col1, met_col2, met_col3 = st.columns(3)
                                        with met_col1:
                                            st.metric("CAM Drift", f"{atk_data['cam_drift']:.3f}")
                                            st.caption("Attention shift magnitude")
                                        with met_col2:
                                            st.metric("CAM Focus", f"{atk_data['cam_focus']:.3f}")
                                            st.caption("Attention concentration")
                                        with met_col3:
                                            st.metric("CAM Entropy", f"{atk_data['cam_entropy']:.3f}")
                                            st.caption("Attention dispersion")
                                else:
                                    st.info("Clean tensor not available for Grad-CAM analysis.")
                                    
                            except Exception as e:
                                st.warning(f"Could not generate Grad-CAM: {e}")
            
            # ============================================================
            # SECTION 5: AGGREGATE RETRAINING GUIDANCE
            # ============================================================
            if show_xai:
                st.divider()
                st.subheader("🛠️ Aggregate Retraining Guidance")
                st.caption("Consolidated recommendations based on all tested images")
                
                st.write("")  # Spacing
                
                # Aggregate all failure types and recommendations across all images
                all_failure_types = set()
                all_recommendations = set()
                has_any_broken = False
                
                for r in all_reports:
                    for atk_name, data in r['attacks'].items():
                        if data['status'] == "Broken":
                            has_any_broken = True
                            failure_types = data.get('failure_types', [])
                            recommendations = data.get('recommendations', [])
                            
                            all_failure_types.update(failure_types)
                            all_recommendations.update(recommendations)
                
                if has_any_broken:
                    # Display aggregated failure types
                    if all_failure_types:
                        st.markdown("**Detected Failure Patterns Across All Images:**")
                        
                        # Check for critical failures
                        has_critical_failure = any(
                            f in all_failure_types 
                            for f in ["Attention Drift", "Patch Dominance", "Diffuse Attention"]
                        )
                        
                        failure_text = "\n".join([f"- {ft}" for ft in sorted(all_failure_types)])
                        
                        if has_critical_failure:
                            st.warning(f"**Critical failures detected:**\n\n{failure_text}")
                        else:
                            st.info(f"**Detected issues:**\n\n{failure_text}")
                        
                        st.write("")  # Spacing
                    
                    # Display aggregated recommendations
                    if all_recommendations:
                        st.markdown("**Recommended Training Improvements:**")
                        st.caption("Apply these techniques to improve model robustness")
                        
                        for rec in sorted(all_recommendations):
                            st.markdown(f"✓ {rec}")
                else:
                    st.success("**Model Robustness Validated**\n\nThe model shows stable behavior under all tested attacks. No immediate retraining required.")
