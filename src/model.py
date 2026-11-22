"""Model: evaluation functions (more functions are added in the next step)."""
import numpy as np


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
