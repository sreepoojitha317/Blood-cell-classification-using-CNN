import tensorflow as tf

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_transfer_model():
    """
    Build a blood cell classification model
    using MobileNetV2 transfer learning.
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(*IMAGE_SIZE, 3)
        ),

        # MobileNetV2 preprocessing
        tf.keras.layers.Rescaling(
            2.0,
            offset=-1.0
        ),

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model