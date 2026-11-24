"""Preprocessing: turning image files into data the model can use."""
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from src.config import BATCH_SIZE, IMG_SIZE

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(folder, class_names):
    """Return two lists: image paths and their label numbers (0, 1, 2).
    Expects the layout:  folder/<class_name>/<image files>"""
    paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in sorted(os.listdir(class_folder)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(class_folder, filename))
                labels.append(label)
    return paths, labels


def load_and_resize(path, label):
    """Read one image file, decode it and resize it to 224 x 224 (values stay 0-255)."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def make_dataset(paths, labels, shuffle=False):
    """Turn lists of paths and labels into batches the model can read."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=42)
    dataset = dataset.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_augmentation():
    """Random changes applied to training images. Only active during training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.2),
    ], name="augmentation")


def validate_image(file_bytes):
    """Check that uploaded bytes are a real image. Returns a PIL image or raises ValueError."""
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise ValueError("The file is not a valid image (use .jpg or .png).")


def image_to_array(pil_image):
    """PIL image -> array of shape (1, 224, 224, 3), resized the same way as in training."""
    array = np.asarray(pil_image, dtype=np.float32)
    array = tf.image.resize(array, IMG_SIZE).numpy()
    return np.expand_dims(array, axis=0)
