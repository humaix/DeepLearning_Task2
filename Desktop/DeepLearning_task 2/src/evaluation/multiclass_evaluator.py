"""
Evaluator for multi-class classification models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix,f1_score,)
from src.evaluation.base_evaluator import BaseEvaluator
from src.utils.logger import get_logger

class MultiClassEvaluator(BaseEvaluator):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.output_directory = Path("outputs/metrics")
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True)
    def evaluate(self, model, x_test, y_test):
        self.logger.info("Multi-class evaluation started.")
        predictions = model.predict(
            x_test,
            verbose=0)
        predicted_labels = np.argmax(
            predictions,
            axis=1)
        actual_labels = np.argmax(
            y_test,
            axis=1)
        accuracy = accuracy_score(
            actual_labels,
            predicted_labels)
        f1 = f1_score(
            actual_labels,
            predicted_labels,
            average="weighted")
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Weighted F1 Score",
                ],
                "Value": [
                    accuracy,
                    f1,
                ],
            }
        )
        metrics.to_csv(
            self.output_directory / "cnn_metrics.csv",
            index=False,)
        report = classification_report(
            actual_labels,
            predicted_labels)
        with open(
            self.output_directory / "cnn_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)
        matrix = confusion_matrix(
            actual_labels,
            predicted_labels)
        self.logger.info("\nConfusion Matrix\n%s", matrix)
        return {
            "accuracy": accuracy,
            "f1_score": f1,
        }