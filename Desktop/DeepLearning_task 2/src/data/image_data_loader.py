"""
Data loader for image datasets.
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from src.utils.logger import get_logger

class ImageDataLoader:
    """
    Loads and preprocesses image datasets.
    """
    def __init__(self, config):
        self.logger = get_logger(__name__)
        self.config = config
    def load_data(self):
        """
        Load and preprocess the MNIST dataset.
        """
        self.logger.info("Loading MNIST dataset.")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.logger.info("Preprocessing images.")
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        x_train = x_train.reshape((-1, 28, 28, 1))
        x_test = x_test.reshape((-1, 28, 28, 1))

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return x_train, x_test, y_train, y_test