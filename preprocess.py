"""
OpenCV-based image preprocessing pipeline.

Pipeline steps:
  1. Read image
  2. Resize to IMG_SIZE x IMG_SIZE
  3. Convert BGR -> RGB
  4. Normalize pixel values to [0, 1]

For training, augmentation is handled by Keras's ImageDataGenerator
in train.py (faster than per-image OpenCV augmentation in tf.data).
"""
import cv2
import numpy as np
from pathlib import Path
from src.config import IMG_SIZE


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """Load a single image from disk and run it through the preprocessing pipeline.

    Returns
    -------
    np.ndarray of shape (IMG_SIZE, IMG_SIZE, 3), dtype float32, range [0, 1].
    Raises FileNotFoundError if the image cannot be read.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Resize (cv2 uses (width, height) order)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_batch(image_paths: list[str | Path]) -> np.ndarray:
    """Preprocess a list of image paths into a single batch tensor."""
    return np.stack([load_and_preprocess(p) for p in image_paths], axis=0)
