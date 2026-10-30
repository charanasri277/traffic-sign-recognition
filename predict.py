"""Predict a traffic sign from an image.

Usage:
    python predict.py path/to/image.png
"""
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 32
MODEL_PATH = os.path.join("models", "traffic_sign_cnn.h5")
CLASS_NAMES_PATH = "class_names.txt"


def load_class_names(path=CLASS_NAMES_PATH):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python train.py' first.")
    return load_model(path, compile=False)


def preprocess_rgb(rgb_image):
    """rgb_image: HxWx3 uint8 array -> (1, 32, 32, 3) float array in 0-1."""
    img = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model, rgb_image, class_names, top_k=3):
    probs = model.predict(preprocess_rgb(rgb_image), verbose=0)[0]
    top = np.argsort(probs)[::-1][:top_k]
    return [(int(i), class_names[i], float(probs[i]) * 100) for i in top]


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model = load_trained_model()
    class_names = load_class_names()
    results = predict_image(model, img, class_names)

    cid, name, conf = results[0]
    print(f"\nPredicted class : {cid} - {name}")
    print(f"Confidence      : {conf:.2f}%")
    print("\nTop 3 predictions:")
    for cid, name, conf in results:
        print(f"  {cid:2d}  {name:<50s} {conf:6.2f}%")


if __name__ == "__main__":
    main()
