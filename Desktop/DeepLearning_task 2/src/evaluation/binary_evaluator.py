"""
Binary classification evaluator.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score,confusion_matrix,)
from src.utils.logger import get_logger

class BinaryEvaluator:
    """
    Evaluates binary classification models.
    """
    def __init__(self):

        self.logger = get_logger(__name__)
    def evaluate(self,
        model,
        x_test=None,
        y_test=None,
        test_dataset=None,):

        self.logger.info("Binary model evaluation started.")


        # Transfer Learning (tf.data.Dataset)
        if test_dataset is not None:
            y_true = []
            y_pred = []
            for images, labels in test_dataset:
                predictions = model.predict(images, verbose=0)
                predictions = (predictions > 0.5).astype(int)
                y_true.extend(labels.numpy())
                y_pred.extend(predictions.flatten())
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

        # ANN / RNN / LSTM
        else:
            predictions = model.predict(
                x_test,
                verbose=0)
            predictions = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(
                y_test,
                predictions)
            cm = confusion_matrix(
                y_test,
                predictions)
        self.logger.info(f"\nConfusion Matrix\n{cm}")

        return {
            "accuracy": accuracy
}