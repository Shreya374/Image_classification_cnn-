# Image Classification with CNN

A deep learning project that classifies images into custom categories using Convolutional Neural Networks. Built primarily with **TensorFlow/Keras**, with **PyTorch** explored for benchmarking and **OpenCV** powering the preprocessing pipeline.

> **Best validation accuracy: 92%+** on a custom 3-class image dataset.

---

## Overview

This project builds an end-to-end image classification pipeline — from raw images to a trained CNN model that achieves over 90% validation accuracy. The goal was to learn how convolutional architectures extract spatial features and to compare classical CNN designs (VGG-style, ResNet-inspired) on a small custom dataset.

**What the model does:** given an input image, it predicts which of 3 categories the image belongs to, along with a confidence score.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras (primary), PyTorch (benchmarking) |
| Image Processing | OpenCV, NumPy |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## Project Structure

```
Image-Classification-CNN/
│
├── data/
│   ├── train/              # Training images organized by class folders
│   ├── val/                # Validation set
│   └── test/               # Held-out test set
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── preprocess.py       # OpenCV preprocessing pipeline
│   ├── model.py            # CNN architecture (Keras)
│   ├── train.py            # Training loop
│   └── evaluate.py         # Metrics and confusion matrix
│
├── models/                 # Saved .h5 / .keras model checkpoints
├── results/                # Plots, confusion matrices, sample predictions
├── requirements.txt
└── README.md
```

---

## Dataset

A **custom 3-class image dataset** collected and curated for this project. Each class folder contains training, validation, and test splits in roughly an 80 / 10 / 10 ratio.

Class distribution was reviewed during EDA to check for imbalance, and light augmentation was applied to the under-represented class.

---

## Preprocessing Pipeline (OpenCV)

All images pass through a consistent preprocessing pipeline before training:

1. **Resize** — uniform resize to `224 x 224` using `cv2.resize`
2. **Color conversion** — BGR → RGB via `cv2.cvtColor`
3. **Normalization** — pixel values scaled to `[0, 1]`
4. **Augmentation** (training set only) — random horizontal flips, rotations (±15°), brightness shifts, and zoom
5. **Batching** — fed to the model via `tf.data.Dataset` for efficient loading

This pipeline ensured the model saw varied versions of each image and helped reduce overfitting on the small dataset.

---

## Model Architecture

The primary model is a **VGG-inspired CNN** built in Keras:

```
Input (224 x 224 x 3)
  → Conv2D(32) → ReLU → MaxPool
  → Conv2D(64) → ReLU → MaxPool
  → Conv2D(128) → ReLU → MaxPool
  → Flatten
  → Dense(128) → ReLU → Dropout(0.5)
  → Dense(3, softmax)
```

**Training setup**

- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam (lr = 1e-3)
- **Batch size:** 32
- **Epochs:** 25 (with early stopping on val loss)
- **Callbacks:** `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

A **ResNet-inspired variant** with skip connections was also tested for comparison.

---

## Results

| Metric | Score |
|---|---|
| Training accuracy | ~95% |
| **Validation accuracy** | **92%+** |
| Test accuracy | ~91% |

**What was visualized:**

- Training vs validation accuracy/loss curves
- Confusion matrix on the test set
- Per-class precision, recall, F1 (via `sklearn.metrics.classification_report`)
- Sample predictions with confidence scores

All plots are saved under `results/`.

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/Shreya374/Image-Classification-CNN.git
cd Image-Classification-CNN
pip install -r requirements.txt
```

### 2. Prepare data

Place your images under `data/train/<class_name>/`, `data/val/<class_name>/`, `data/test/<class_name>/`.

### 3. Train

```bash
python src/train.py --epochs 25 --batch_size 32
```

### 4. Evaluate

```bash
python src/evaluate.py --model models/best_model.keras
```

### 5. Predict on a single image

```bash
python src/predict.py --image path/to/image.jpg
```

---

## Key Learnings

- **Data quality beats model complexity** on small datasets — clean preprocessing and good augmentation moved validation accuracy more than swapping architectures did.
- **Early stopping mattered** — the model started overfitting around epoch 18; without callbacks, validation accuracy degraded.
- **TensorFlow vs PyTorch** — building a parallel PyTorch version made the underlying mechanics (autograd, training loops, data loaders) much clearer compared to Keras's higher-level API.
- **OpenCV + tf.data** is a fast, lightweight pipeline for small to mid-size image datasets without needing heavy frameworks.

---

## Future Work

- Try **transfer learning** with pretrained MobileNetV2 / EfficientNet backbones
- Experiment with **model quantization** to shrink size for edge deployment
- Wrap the model in a **Flask/FastAPI** endpoint for real-time inference
- Build a small **Streamlit** demo for uploading images and seeing predictions

---

## Author

**Shreya Pandurang Jagtap**
B.Tech, Computer Science Engineering — D.Y. Patil Technical Campus, Kolhapur

[GitHub](https://github.com/Shreya374) • [LinkedIn](https://www.linkedin.com/in/shreya-jagtap) • shreyajagtap374@gmail.com

---

## License

MIT — feel free to use this project as a learning reference.
