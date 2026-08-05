"""
Utility class for plotting training results.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
class Visualizer:
    """
    Creates and saves training plots.
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.output_directory = Path("outputs/plots")
        self.output_directory.mkdir(parents=True, exist_ok=True)
    def plot_history(self, history):
        """
        Save training accuracy and loss graphs.
        """
        self.logger.info("Generating training plots.")
        # Accuracy Plot
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_directory / "accuracy.png")
        plt.close()

        # Loss Plot
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_directory / "loss.png")
        plt.close()
        self.logger.info("Training plots saved successfully.")