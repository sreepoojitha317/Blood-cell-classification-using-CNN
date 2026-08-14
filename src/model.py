import tensorflow as tf

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(*IMAGE_SIZE, 3)
        ),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation="relu"
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES, activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model