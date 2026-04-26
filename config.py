"""
Central configuration for the image classification project.
Update CLASS_NAMES and DATA_DIR to match your dataset.
"""
from pathlib import Path

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Make sure output dirs exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- dataset ----------
# Update these to match your folder names under data/train/
CLASS_NAMES = ["class_a", "class_b", "class_c"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------- image / training ----------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
SEED = 42
