import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import EPOCHS, MODEL_DIR, RESULTS_DIR
from src.data_loader import load_datasets
from src.transfer_model import build_transfer_model


def plot_history(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("MobileNetV2 Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mobilenet_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("MobileNetV2 Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mobilenet_loss.png")
    plt.close()


def train_transfer_model():

    print("Loading datasets...")

    train_dataset, validation_dataset, test_dataset = load_datasets()

    print("Building MobileNetV2 model...")

    model = build_transfer_model()

    print("\nStarting MobileNetV2 training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    print("\nEvaluating MobileNetV2 on test data...")

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "blood_cell_mobilenet.keras"

    model.save(model_path)

    print(f"\nMobileNetV2 model saved to: {model_path}")

    plot_history(history)

    print(f"Training graphs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    train_transfer_model()