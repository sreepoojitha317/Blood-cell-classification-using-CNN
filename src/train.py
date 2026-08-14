import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import EPOCHS, MODEL_DIR, RESULTS_DIR
from src.data_loader import load_datasets
from src.model import build_cnn_model


def plot_training_history(history):
    """Save training and validation accuracy/loss plots."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png")
    plt.close()


def train_model():
    """Load data, train CNN, evaluate and save the model."""

    print("Loading datasets...")
    train_dataset, test_dataset = load_datasets()

    print("Building CNN model...")
    model = build_cnn_model()

    model.summary()

    # Use part of training data for validation
    validation_size = 0.2
    total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    validation_batches = int(total_batches * validation_size)

    validation_dataset = train_dataset.take(validation_batches)
    training_dataset = train_dataset.skip(validation_batches)

    print("\nStarting training...")

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    # Evaluate on the separate TEST dataset
    print("\nEvaluating on test data...")

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "blood_cell_cnn.keras"
    model.save(model_path)

    print(f"\nModel saved to: {model_path}")

    # Save graphs
    plot_training_history(history)

    print(f"Training graphs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    train_model()