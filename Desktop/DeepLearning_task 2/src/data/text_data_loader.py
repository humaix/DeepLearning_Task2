"""
Data loader for text datasets.
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.logger import get_logger

class TextDataLoader:
    """
    Loads and preprocesses text datasets.
    """
    def __init__(self,config):
        self.logger = get_logger(__name__)
        self.config = config
    def load_data(self):
        """
        Load and preprocess the IMDB movie review dataset.
        """
        self.logger.info("Loading IMDB dataset.")
        vocabulary_size = 10000
        max_length = 200
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            num_words=vocabulary_size)

        self.logger.info("Padding text sequences.")
        x_train = pad_sequences(
            x_train,
            maxlen=max_length)

        x_test = pad_sequences(x_test,maxlen=max_length)

        return ( x_train, x_test,y_train,y_test,)