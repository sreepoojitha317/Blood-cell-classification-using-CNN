import numpy as np
import tensorflow as tf

from src.config import IMAGE_SIZE, CLASS_NAMES, MODEL_DIR


# ============================================================
# Load trained CNN V2 model
# ============================================================

MODEL_PATH = MODEL_DIR / "blood_cell_cnn_v2_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# Prediction Function
# ============================================================

def predict_image(image_path):
    """
    Predict the blood cell type from an image.

    Returns:
        predicted_class: Predicted blood cell class
        confidence: Prediction confidence percentage
    """

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # Dataset images are already scaled to [0, 1],
    # so scale the uploaded image in the same way.
    image_array = image_array / 255.0

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    # Get predicted class index
    predicted_index = np.argmax(predictions[0])

    # Get class name
    predicted_class = CLASS_NAMES[predicted_index]

    # Get confidence
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence