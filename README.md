# Real-Time Human Emotion Detection Pipeline

A two-stage pipeline that detects human faces in real time and classifies their emotion using a custom-trained classifier built from scratch — no scikit-learn, no pre-built classifiers.

---

## Project Goal

Detect and classify facial emotions from a live webcam feed into one of **7 classes**:

> Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral

---

## Architecture

```
Webcam Frame
     │
     ▼
┌─────────────────────────┐
│  Stage 1: Face Detection │  YOLOv11n (Hugging Face)
│  AdamCodd/YOLOv11n-face  │  → Bounding box coords
└─────────────────────────┘
     │
     ▼  Crop + Resize to 224×224 + Normalize
     │
┌──────────────────────────────┐
│  Stage 2a: Feature Extraction │  ViT-B/16 (google/vit-base-patch16-224-in21k)
│  CLS token → 768-dim vector  │
└──────────────────────────────┘
     │
     ▼
┌───────────────────────────────────┐
│  Stage 2b: Emotion Classification  │  Custom Logistic Regression
│  W (768×7) + b (7,) → Softmax     │  trained with Newton's Method (L-BFGS)
└───────────────────────────────────┘
     │
     ▼
Emotion Label + Confidence overlaid on frame
```

---

## File Overview

| File | Purpose |
|------|---------|
| `FaceDetection.py` | Main entry point — runs the live webcam pipeline |
| `Feature_Extracting.py` | `ViTFeatureExtractor` class — batch-extract 768-dim feature vectors from images |
| `prepare_data.py` | Merge emotion CSVs, stratified split into 5 train/val/test iterations |
| `Training.py` | Train, evaluate, and visualize the emotion classifier |
| `NewtonMethod.py` | `CustomLogisticRegression` — PyTorch implementation using L-BFGS optimizer |
| `pipeline.py` | Generate a Graphviz architecture diagram (`pipeline_architecture.png`) |

---

## Setup

### Requirements

```bash
pip install torch torchvision transformers ultralytics huggingface_hub \
            opencv-python pillow pandas numpy matplotlib graphviz
```

Graphviz binary (for `pipeline.py`): https://graphviz.org/download/

### Hardware

Runs on CPU or CUDA GPU. ViT inference is noticeably faster on GPU.

---

## Usage

### 1. Extract Features from Your Dataset

Place emotion images in a folder structure and run:

```bash
python Feature_Extracting.py
```

This produces per-emotion CSV files (e.g. `vit_happy_features.csv`) with 768-column feature vectors.

### 2. Prepare Training Data

```bash
python prepare_data.py
```

Merges the emotion CSVs, shuffles, and creates 5 stratified splits:

```
iteration_1/
    train_features.csv   # 70%
    val_features.csv     # 15%
    test_features.csv    # 15%
iteration_2/ ...
```

### 3. Train the Classifier

```bash
python Training.py
```

Trains `CustomLogisticRegression` on each iteration using L-BFGS. Saves:
- `models/emotion_model_iter{n}.pth` — trained weights
- `results_iter{n}/` — confusion matrices, training history, accuracy comparison charts

### 4. Run Live Emotion Detection

```bash
python FaceDetection.py
```

Opens your default webcam. Press **`q`** to quit.

The pipeline loads:
- YOLOv11n face detector (auto-downloaded from Hugging Face)
- ViT-B/16 feature extractor
- Trained model from `models/emotion_model_iter2.pth`

---

## Model Details

### Feature Extractor — ViT-B/16

- Model: `google/vit-base-patch16-224-in21k`
- Input: 224×224 RGB image
- Output: 768-dimensional CLS token embedding
- Weights are frozen — used purely for feature extraction

### Classifier — Custom Logistic Regression

- Implemented from scratch in PyTorch (no `nn.Module`)
- Parameters: weight matrix `W` (768×7) and bias `b` (7,)
- Optimizer: L-BFGS (Newton's Method) via `torch.optim.LBFGS`
- Loss: Cross-Entropy
- All evaluation metrics (accuracy, precision, recall, F1, confusion matrix) implemented manually

### Data Splits

- 70% training / 15% validation / 15% test
- Stratified splitting to preserve class balance across 5 random iterations

---

## Built With

- **Face Detection:** YOLOv11 (Ultralytics) via Hugging Face
- **Feature Extraction:** Vision Transformer ViT-B/16 (HuggingFace Transformers)
- **Classification:** Custom PyTorch logistic regression + L-BFGS
- **Image Processing:** OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn (optional), Graphviz
