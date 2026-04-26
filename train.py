"""
Train the CNN classifier.

Usage:
    python -m src.train
    python -m src.train --epochs 30 --batch_size 16
"""
import argparse
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from src.config import (
    TRAIN_DIR, VAL_DIR, MODELS_DIR, RESULTS_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, NUM_CLASSES, CLASS_NAMES
)
from src.model import build_cnn


def get_data_generators(batch_size: int):
    """Create augmented training generator and clean validation generator."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
    )
    val_flow = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_flow, val_flow


def plot_history(history, out_path):
    """Save training/validation accuracy + loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved training curves -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    tf.random.set_seed(SEED)

    # Data
    train_flow, val_flow = get_data_generators(args.batch_size)
    print(f"Found {train_flow.samples} train and {val_flow.samples} val images")
    print(f"Class indices: {train_flow.class_indices}")

    # Model
    model = build_cnn()
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    ckpt_path = MODELS_DIR / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # Train
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save artifacts
    plot_history(history, RESULTS_DIR / "training_curves.png")
    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"\nBest model saved -> {ckpt_path}")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
