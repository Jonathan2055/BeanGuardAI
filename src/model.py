"""Model: building, fine-tuning settings and evaluation."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.config import IMG_SIZE
from src.preprocessing import build_augmentation


def build_model(num_classes, pretrained=True):
    """MobileNetV2 (pre-trained on ImageNet) + a small new classification head.
    The model takes raw images (0-255): rescaling and augmentation are inside the model."""
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,                                  # remove ImageNet's 1000-class output
        weights="imagenet" if pretrained else None,
    )
    base.trainable = False                                  # phase 1: keep the pre-trained knowledge

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = build_augmentation()(inputs)                        # only active while training
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)           # 0-255 -> -1...1 (what MobileNetV2 expects)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    return keras.Model(inputs, outputs, name="bean_leaf_classifier")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def find_base(model):
    """Return the MobileNetV2 part of the model."""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("MobileNetV2 base not found in the model")


def unfreeze_top_layers(model, n_layers=30):
    """Fine-tuning: let the last n layers of MobileNetV2 learn (BatchNormalization stays frozen)."""
    base = find_base(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def predict_all(model, dataset):
    """Predict every image in a dataset. Returns the true labels and the predicted labels."""
    y_true = np.concatenate([labels.numpy() for images, labels in dataset])
    y_pred = np.argmax(model.predict(dataset, verbose=0), axis=1)
    return y_true, y_pred


def evaluate_model(model, dataset, class_names):
    """Loss, accuracy, macro precision / recall / F1, per-class F1 and the confusion matrix."""
    loss, accuracy = model.evaluate(dataset, verbose=0)
    y_true, y_pred = predict_all(model, dataset)

    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)       # rows = true class, columns = predicted class
    for true, predicted in zip(y_true, y_pred):
        matrix[true, predicted] += 1

    true_positives = np.diag(matrix)
    predicted_totals = matrix.sum(axis=0)
    actual_totals = matrix.sum(axis=1)
    precision = np.divide(true_positives, predicted_totals, out=np.zeros(n), where=predicted_totals > 0)
    recall = np.divide(true_positives, actual_totals, out=np.zeros(n), where=actual_totals > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros(n), where=both > 0)

    return {
        "loss": round(float(loss), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision.mean()), 4),
        "recall": round(float(recall.mean()), 4),
        "f1": round(float(f1.mean()), 4),
        "per_class_f1": {name: round(float(v), 4) for name, v in zip(class_names, f1)},
        "confusion_matrix": matrix.tolist(),
        "images": int(len(y_true)),
    }


def export_inference_model(trained_model):
    """The same model WITHOUT the augmentation block (augmentation is only needed while training).
    Used to create the .h5 file: the RandomBrightness layer cannot be saved in .h5 format."""
    inputs = keras.Input(shape=trained_model.input_shape[1:])
    x = inputs
    for layer in trained_model.layers[2:]:        # skip the InputLayer and the augmentation block
        x = layer(x)
    return keras.Model(inputs, x, name="bean_leaf_inference")
