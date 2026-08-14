import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES
from src.data_loader import load_datasets


MODEL_PATH = "models/blood_cell_cnn_v2_best.keras"


def evaluate_model():

    print("Loading datasets...")
    train_dataset, validation_dataset, test_dataset = load_datasets()

    print("\nLoading best CNN V2 model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Collect true labels
    y_true = []
    y_pred = []

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    # Confusion Matrix
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)


if __name__ == "__main__":
    evaluate_model()