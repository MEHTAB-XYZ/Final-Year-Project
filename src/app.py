# src/app.py

import streamlit as st
from PIL import Image
from loader import load_model
from inference import run_inference
from utils import class_name, tensor_to_pil

st.set_page_config(layout="wide")
st.title("AutoRobustXAI – Adversarial Robustness Analyzer")

model = load_model()

uploaded = st.file_uploader("Upload a Traffic Sign Image", type=["jpg","png","jpeg"])

attack = st.selectbox(
    "Choose Attack",
    ["none", "fgsm", "pgd", "bim", "fog", "rain", "brightness"]
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    result = run_inference(model, pil_img, attack)

    # ---------------------------------------------------------
    # SIDE BY SIDE IMAGE DISPLAY
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(pil_img, width=300)

    with col2:
        st.subheader("Attacked Image")

        if attack in ["fgsm", "pgd", "bim"]:
            attacked_img = tensor_to_pil(result["adv_tensor"])
        else:
            attacked_img = result["adv_pil"]

        st.image(attacked_img, width=300)

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------
    st.subheader("Predictions")
    st.write(f"Original Prediction: {class_name(result['orig_label'])} ({result['orig_conf']:.4f})")
    st.write(f"Adversarial Prediction: {class_name(result['adv_label'])} ({result['adv_conf']:.4f})")

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    st.subheader("Metrics")
    st.write(f"Confidence Drop: {result['confidence_drop']:.4f}")
    st.write(f"Attack Success: {result['attack_success']}")
    st.write(f"Robustness Score: {result['robustness_score']:.4f}")

    # ---------------------------------------------------------
    # GRAD-CAM
    # ---------------------------------------------------------
    st.subheader("Grad-CAM Visualization")
    st.image(result["gradcam"], width=400)
