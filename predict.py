"""
Predict the class of a single image using a trained model.

Usage:
    python -m src.predict --image path/to/image.jpg
    python -m src.predict --image path/to/image.jpg --model models/best_model.keras
"""
import argparse
import numpy as np
import tensorflow as tf

from src.config import MODELS_DIR, CLASS_NAMES
from src.preprocess import load_and_preprocess


def predict_image(image_path: str, model_path: str) -> dict:
    """Run prediction on a single image and return top class + all probs."""
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)

    probs = model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "best_model.keras"))
    args = parser.parse_args()

    result = predict_image(args.image, args.model)

    print(f"\nImage:           {args.image}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence:      {result['confidence']:.4f}")
    print("\nAll class probabilities:")
    for cls, p in sorted(result["all_probabilities"].items(),
                         key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:15s} {p:.4f}  {bar}")


if __name__ == "__main__":
    main()
