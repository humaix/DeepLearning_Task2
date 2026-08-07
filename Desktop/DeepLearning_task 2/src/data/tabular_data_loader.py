"""
Data loader for tabular datasets.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

class TabularDataLoader:
    """
    Loads and preprocesses tabular datasets.
    """
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load the Breast Cancer dataset.
        """

        self.logger.info("Loading Breast Cancer dataset.")

        dataset = load_breast_cancer()

        x = dataset.data
        y = dataset.target

        return x, y
    def preprocess_data(self, x, y, test_size=0.2, random_state=42):
        """
        Split and scale the dataset.
        """

        self.logger.info("Splitting dataset.")

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y)

        self.logger.info("Scaling features.")
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        return x_train, x_test, y_train, y_test