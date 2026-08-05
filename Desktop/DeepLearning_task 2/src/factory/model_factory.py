"""
Factory class for creating deep learning models.
"""

from src.models.ann_model import ANNModel
from src.models.cnn_model import CNNModel
from src.models.rnn_model import RNNModel
from src.models.lstm_model import LSTMModel
from src.models.transfer_model import TransferModel
class ModelFactory:
    """
    Creates model instances based on configuration.
    """
    @staticmethod
    def create_model(config):
        model_name = config["model_name"].upper()
        models = {
            "ANN": ANNModel,
            "CNN": CNNModel,
            "RNN": RNNModel,
            "LSTM": LSTMModel,
            "TRANSFER": TransferModel,
        }
        if model_name not in models:
            raise ValueError(
                f"Unsupported model: {model_name}")
        return models[model_name](config)