"""
Evaluate a trained model on the held-out test set.

Outputs:
  - Test accuracy / loss to stdout
  - Per-class precision/recall/F1 (sklearn classification report)
  - Confusion matrix plot saved to results/

Usage:
    python -m src.evaluate
    python -m src.evaluate --model models/best_model.keras
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    TEST_DIR, MODELS_DIR, RESULTS_DIR,
    IMG_SIZE, BATCH_SIZE, CLASS_NAMES
)


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "best_model.keras"))
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_flow = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    # Overall metrics
    loss, acc = model.evaluate(test_flow, verbose=1)
    print(f"\nTest loss:     {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    # Per-class metrics
    test_flow.reset()
    y_pred_probs = model.predict(test_flow, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_flow.classes

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, RESULTS_DIR / "confusion_matrix.png")

    # Save the report to a text file too
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test loss:     {loss:.4f}\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
