# OZAIANet: Thermal Image-Based Diabetes Classification

Streamlit web application for demonstrating **OZAIANet**, a deep-learning model designed for diabetic microvascular pattern evaluation from plantar foot thermograms.

---

## 🌟 Overview

OZAIANet utilizes a dual-backbone convolutional neural network:
- **DenseNet121**
- **EfficientNetV2B0**

The model takes as input segmented thermal images of the sole of the foot and computes the probability of diabetic-associated thermal alterations.

### Model Specifications & Pipeline
- **Input Dimensions**: `224 × 224 × 3` (RGB)
- **Pixel Normalization**: Scaled to `[0.0, 1.0]` by dividing by `255.0`
- **Model Output**: Sigmoid probability value $p \in [0.0, 1.0]$
- **Classification Threshold**: `0.50`
  - $p > 0.50 \implies$ **DM** (Diabetic Pattern)
  - $p \le 0.50 \implies$ **CTRL** (Control / Non-Diabetic Pattern)

---

## 📁 Project Structure

```text
ozai-net-app/
├── app.py                     # Main Streamlit web application
├── utils.py                   # Image validation safety gate & inference pipeline
├── best_model_2.h5            # Trained OZAIANet weights (Keras/TensorFlow)
├── predict.py                 # Standalone script for quick local CLI testing
├── test_model.py              # Quick script to verify model loading & tensor shapes
├── samples/                   # Benchmark plantar thermal scans
│   ├── pic1.png               # Sample 1 (DM)
│   ├── pic2.png               # Sample 2 (DM)
│   ├── pic3.png               # Sample 3 (CTRL)
│   ├── pic4.png               # Sample 4 (CTRL)
│   ├── pic5.png               # Sample 5 (CTRL)
│   └── pic6.png               # Sample 6 (DM)
├── requirements.txt           # Minimal production dependencies
└── README.md                  # Documentation and usage guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ (tested on Python 3.12) is installed.

### 2. Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Launching the Application
Run the Streamlit app:

```bash
streamlit run app.py
```

The web interface will open in your default browser at `http://localhost:8501`.

---

## 🛡️ Image Validation / Safety Gate

The OZAIANet model was specifically trained on segmented plantar thermograms of the sole of the foot. Sending arbitrary photos (such as landscapes, faces, everyday objects, or general medical scans) produces meaningless predictions.

To protect model integrity, `app.py` implements a **modular pre-inference safety gate** (`validate_thermal_foot_image` in `utils.py`).

### Verification Checks
1. **Readable Format & Resolution**: Confirms image is decodeable, has minimum dimensions ($\ge 50 \times 50$), and a reasonable crop aspect ratio ($\le 2.5:1$).
2. **Segmented Dark Background**: Verifies that the image contains a dark background region ($12\% - 88\%$) characteristic of thermal scans.
3. **Dark Perimeter Margins**: Checks that the outer $5\%$ border of the image is predominantly dark background ($\ge 35\%$), ensuring the foot is framed properly.
4. **Single Dominant Object**: Analyzes foreground contours to ensure there is a single primary connected component ($\ge 55\%$ of total foreground) rather than cluttered natural scenes.
5. **Thermal False-Color Palette**: Measures channel standard deviation across RGB channels ($\ge 22.0$) to confirm the presence of false-color thermal mappings (e.g. Ironbow/Jet) rather than monochrome or standard desaturated photos.
6. **Contour Solidity**: Ensures the foreground shape is cohesive ($\text{solidity} \ge 0.65$) matching plantar foot silhouettes.

> **Note on Limitations:** These computer vision heuristics are lightweight conservative filters designed to intercept invalid uploads and casual errors. They do not substitute for rigorous clinical sensor validation or professional thermographic calibration.

---

## 🧪 Benchmark Samples

The application includes 6 reference samples in `samples/` for immediate testing:
- **pic1.png**: DM (Raw Probability: `0.9837`)
- **pic2.png**: DM (Raw Probability: `1.0000`)
- **pic3.png**: CTRL (Raw Probability: `0.1984`)
- **pic4.png**: CTRL (Raw Probability: `0.0018`)
- **pic5.png**: CTRL (Raw Probability: `0.0006`)
- **pic6.png**: DM (Raw Probability: `0.8373`)

---

## ⚠️ Research Disclaimer

This tool is created for **research and demonstration purposes only**. It does not constitute a medical device, diagnostic test, or clinical recommendation.
