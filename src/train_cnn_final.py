import matplotlib.pyplot as plt

from src.config import EPOCHS, MODEL_DIR, RESULTS_DIR
from src.data_loader import load_datasets
from src.model import build_cnn_model


def plot_history(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_cnn_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_cnn_loss.png")
    plt.close()


def train_final_cnn():

    print("Loading datasets...")

    train_dataset, validation_dataset, test_dataset = load_datasets()

    # Cache training and validation data in memory
    print("Caching datasets...")

    train_dataset = train_dataset.cache()
    validation_dataset = validation_dataset.cache()

    print("Building CNN model...")

    model = build_cnn_model()

    print("\nStarting final CNN training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    print("\nEvaluating final CNN on test data...")

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "blood_cell_cnn_final.keras"

    model.save(model_path)

    print(f"\nFinal CNN saved to: {model_path}")

    plot_history(history)

    print(f"Training graphs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    train_final_cnn()