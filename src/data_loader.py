import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES
)


def create_file_lists():
    """Create a stratified train/validation file split."""

    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = Path(TRAIN_DIR) / class_name

        for image_path in class_dir.glob("*"):
            if image_path.is_file():
                image_paths.append(str(image_path))
                labels.append(label)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels
    )


def load_image(path, label):
    """Read and preprocess an image."""

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


def create_dataset(paths, labels, shuffle=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def load_datasets():

    train_paths, train_labels, val_paths, val_labels = (
        create_file_lists()
    )

    # Test dataset
    test_paths = []
    test_labels = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = Path(TEST_DIR) / class_name

        for image_path in class_dir.glob("*"):
            if image_path.is_file():
                test_paths.append(str(image_path))
                test_labels.append(label)

    train_dataset = create_dataset(
        train_paths,
        train_labels,
        shuffle=True
    )

    validation_dataset = create_dataset(
        val_paths,
        val_labels,
        shuffle=False
    )

    test_dataset = create_dataset(
        test_paths,
        test_labels,
        shuffle=False
    )

    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )