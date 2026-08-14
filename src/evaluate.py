import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from src.config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    MODEL_DIR,
    RESULTS_DIR
)


def load_test_dataset():
    """Load the test dataset without normalization."""

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    test_dataset = test_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        )
    )

    return test_dataset


def evaluate_model():

    print("Loading trained model...")

    model_path = MODEL_DIR / "blood_cell_cnn.keras"
    model = tf.keras.models.load_model(model_path)

    print("Loading test dataset...")

    test_dataset = load_test_dataset()

    y_true = []
    y_pred = []

    print("Generating predictions...")

    for images, labels in test_dataset:

        predictions = model.predict(images, verbose=0)

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w"
    ) as file:

        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("Blood Cell Classification - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png"
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print(RESULTS_DIR / "confusion_matrix.png")

    print("\nClassification report saved to:")
    print(RESULTS_DIR / "classification_report.txt")


if __name__ == "__main__":
    evaluate_model()