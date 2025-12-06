# AutoRobustXAI — MVP

Minimal working prototype for AutoRobustXAI (MVP):
- Upload PyTorch (.pt/.pth) or ONNX model
- Upload an image (traffic sign / sample)
- Run FGSM or PGD attacks
- Visualize Grad-CAM before/after
- Compute simple robustness metrics (accuracy drop, attack success, heatmap drift)
- Streamlit dashboard for interactive experimentation

## Setup (Linux / macOS / Windows WSL)
1. Create venv and activate:
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
