"""Train a CNN on GTSRB.

Expected dataset layout (Kaggle "GTSRB - German Traffic Sign Recognition Benchmark"):
    dataset/Train/0/*.png ... dataset/Train/42/*.png

Usage:
    python train.py                        # full dataset
    python train.py --max-per-class 300 --epochs 8   # FAST mode for quick demo
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # save plots to files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 32
NUM_CLASSES = 43
DATASET_DIR = os.path.join("dataset", "Train")
MODEL_PATH = os.path.join("models", "traffic_sign_cnn.h5")
CLASS_NAMES_PATH = "class_names.txt"

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


def find_train_dir():
    """Find the folder that contains the class sub-folders (0..42 or 00000..00042).
    Works for the Kaggle layout and the official GTSRB zip layout."""
    candidates = [
        os.path.join("dataset", "Train"),
        os.path.join("dataset", "GTSRB", "Final_Training", "Images"),
        os.path.join("dataset", "Final_Training", "Images"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # last resort: search under dataset/ for a folder holding many numeric sub-folders
    for root, dirs, _ in os.walk("dataset"):
        if len([d for d in dirs if d.isdigit()]) >= NUM_CLASSES:
            return root
    raise SystemExit(
        "\nDataset not found inside 'dataset/'.\n"
        "Run:  python download_dataset.py   (or follow README 'Dataset setup')\n"
        "Expected: dataset/Train/0/, dataset/Train/1/, ... dataset/Train/42/"
    )


def load_dataset(max_per_class=None):
    train_dir = find_train_dir()
    print("Using dataset folder:", train_dir)
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        folder = None
        for name in (str(class_id), f"{class_id:05d}"):
            p = os.path.join(train_dir, name)
            if os.path.isdir(p):
                folder = p
                break
        if folder is None:
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".png", ".ppm", ".jpg", ".jpeg")))
        if max_per_class:
            files = files[:max_per_class]
        for name in files:
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_id)
        print(f"Loaded class {class_id:2d}: {len(files)} files")
    X = np.array(images, dtype="float32") / 255.0  # normalize to 0-1
    y = np.array(labels, dtype="int32")
    return X, y


def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig("accuracy_graph.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig("loss_graph.png", dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="limit images per class for a faster run")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    X, y = load_dataset(args.max_per_class)
    print("Data shape:", X.shape, "Labels:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    print(f"\nTraining accuracy   : {history.history['accuracy'][-1]:.4f}")
    print(f"Validation accuracy : {history.history['val_accuracy'][-1]:.4f}")
    print(f"Training loss       : {history.history['loss'][-1]:.4f}")
    print(f"Validation loss     : {history.history['val_loss'][-1]:.4f}")
    plot_history(history)
    print("Saved accuracy_graph.png and loss_graph.png")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    present = sorted(set(y_val) | set(y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred, labels=present,
        target_names=[CLASS_NAMES[i] for i in present], zero_division=0))

    cm = confusion_matrix(y_val, y_pred, labels=present)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=present, columns=present).to_string())

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150); plt.close()
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
