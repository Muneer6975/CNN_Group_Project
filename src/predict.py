"""Predict whether an image contains one person or multiple people."""
from pathlib import Path
import sys
import json
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "best_model.keras"
CLASS_PATH = ROOT / "outputs" / "class_names.json"
IMAGE_SIZE = (180, 180)

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/image.jpg")
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: run python src/train.py")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    probabilities = model.predict(array, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    print(f"Prediction: {class_names[best_index]}")
    print(f"Model confidence: {probabilities[best_index] * 100:.2f}%")
    print("Confidence is not a guarantee of correctness.")

if __name__ == "__main__":
    main()
