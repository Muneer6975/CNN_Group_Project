"""
CNN-Based Classification of Images Containing One Person or Multiple People
Beginner-friendly TensorFlow/Keras implementation.

Dataset folders:
dataset/
  train/
    single_person/      # images containing exactly one visible person
    multiple_people/    # images containing at least two visible people
  validation/
    single_person/
    multiple_people/
  test/
    single_person/
    multiple_people/

Run from the project root:
    python src/train.py
Optional single-image prediction after training:
    python src/predict.py path/to/image.jpg
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
CLASS_NAMES = ["multiple_people", "single_person"]  # alphabetical order from directory names

def load_split(split_name):
    folder = DATA / split_name
    if not folder.exists():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split_name == "train"),
        seed=SEED
    )

def build_model():
    # A compact CNN suitable for a beginner coursework demonstration.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    train_ds = load_split("train")
    val_ds = load_split("validation")
    test_ds = load_split("test")

    # Improve input-pipeline throughput without requiring custom data code.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUT / "best_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(OUT / "training_log.csv"))
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model and class order.
    model.save(OUT / "person_classifier.keras")
    (OUT / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
    )

    # Training curves
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    plt.figure()
    plt.plot(epochs_range, history.history["accuracy"], label="Training accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs_range, history.history["loss"], label="Training loss")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "loss_curve.png", dpi=200)
    plt.close()

    # Evaluate once on the held-out test set.
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(values_format="d")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=200)
    plt.close()

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }
    (OUT / "test_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Full results saved to outputs/test_results.json")
    print("Do not report a performance value until this script has run on your dataset.")

if __name__ == "__main__":
    main()
