"""
Evaluation utilities for classification models.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger


class Evaluator:
    """
    Evaluates trained classification models.
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.output_directory = Path("outputs/metrics")
        self.output_directory.mkdir(parents=True, exist_ok=True)
    def evaluate(self, model, x_test, y_test):
        """
        Evaluate model performance.
        """
        self.logger.info("Model evaluation started.")
        predictions = model.predict(x_test, verbose=0)
        predicted_labels = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels)
        recall = recall_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        self.logger.info(f"Accuracy : {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall   : {recall:.4f}")
        self.logger.info(f"F1 Score : {f1:.4f}")
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",],
                "Value": [
                    accuracy,
                    precision,
                    recall,
                    f1,],
            })

        metrics.to_csv(self.output_directory / "ann_metrics.csv",
            index=False,)

        report = classification_report(y_test, predicted_labels)
        with open(
            self.output_directory / "classification_report.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)
        matrix = confusion_matrix(y_test, predicted_labels)
        self.logger.info("\nConfusion Matrix\n%s", matrix)
        self.logger.info("Evaluation completed.")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }