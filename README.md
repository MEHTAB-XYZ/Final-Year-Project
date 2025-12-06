# Auto Robust XAI

**Automated Robustness Testing and Explainability Analysis for Deep Learning Models**

## 🎯 Overview

Auto Robust XAI is a comprehensive framework for evaluating the robustness of deep learning models against adversarial attacks and providing explainability insights. This tool is particularly useful for testing computer vision models on datasets like CIFAR-10, MNIST, and GTSRB (German Traffic Sign Recognition Benchmark).

## 📁 Project Structure

```
auto_robust_xai/
├─ data/                      # Sample datasets (GTSRB / small example)
├─ models/                    # Saved models (.pt / .onnx) for testing
├─ src/
│  ├─ attacks.py              # Wrappers for FGSM, PGD, etc.
│  ├─ explainers.py           # Grad-CAM, SHAP, LIME helpers
│  ├─ metrics.py              # Accuracy drop, attack success, drift
│  ├─ loader.py               # Model + dataset loading utilities
│  ├─ inference.py            # Inference pipelines (clean + adv)
│  ├─ app.py                  # Streamlit dashboard
│  └─ utils.py                # Misc helpers (preprocess, save images)
├─ notebooks/                 # Experiments & demos
├─ reports/                   # Exported PDF/CSV results
├─ requirements.txt
└─ README.md
```

## 🚀 Features

- **Adversarial Attacks**: FGSM, PGD, and more
- **Explainability Methods**: Grad-CAM, SHAP, LIME
- **Robustness Metrics**: Accuracy drop, attack success rate, prediction drift
- **Interactive Dashboard**: Streamlit-based UI for easy experimentation
- **Multiple Datasets**: Support for CIFAR-10, MNIST, GTSRB
- **Model Formats**: PyTorch (.pt, .pth) and ONNX support

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd auto_robust_xai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Streamlit Dashboard

```bash
cd src
streamlit run app.py
```

### Using the Python API

```python
import torch
from src.loader import load_model, get_dataset_loader
from src.attacks import FGSM, PGD
from src.inference import RobustnessEvaluator

# Load model and dataset
model = load_model('models/model.pt')
train_loader, test_loader = get_dataset_loader('cifar10')

# Create attacks
attacks = {
    'FGSM': FGSM(model, epsilon=0.03),
    'PGD': PGD(model, epsilon=0.03, alpha=0.01, num_iter=10)
}

# Evaluate robustness
evaluator = RobustnessEvaluator(model, attacks)
results = evaluator.evaluate(test_loader, max_batches=10)

print(results)
```

### Generating Grad-CAM Visualizations

```python
from src.explainers import GradCAM
from src.utils import save_image

# Assuming you have a model with a target layer
target_layer = model.layer4[-1]  # Example for ResNet
gradcam = GradCAM(model, target_layer)

# Generate CAM
heatmap = gradcam.generate_cam(input_image)
save_image(heatmap, 'gradcam_output.png')
```

## 📊 Metrics

The framework provides several metrics to evaluate model robustness:

- **Clean Accuracy**: Model accuracy on original images
- **Adversarial Accuracy**: Model accuracy on adversarial images
- **Accuracy Drop**: Difference between clean and adversarial accuracy
- **Attack Success Rate**: Percentage of successful adversarial attacks
- **Prediction Drift**: L2 distance between clean and adversarial predictions
- **Confidence Drop**: Change in model confidence due to attacks

## 🔬 Supported Attacks

- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)
- More attacks can be easily added by extending the `attacks.py` module

## 🔍 Explainability Methods

- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **SHAP**: SHapley Additive exPlanations (requires `shap` library)
- **LIME**: Local Interpretable Model-agnostic Explanations (requires `lime` library)

## 📝 Example Notebooks

Check the `notebooks/` directory for example Jupyter notebooks demonstrating:
- Model training and evaluation
- Adversarial attack generation
- Explainability visualization
- Robustness analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the interactive dashboard framework
- Research papers on adversarial robustness and explainability

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research and educational tool. Always validate results and use responsibly.
