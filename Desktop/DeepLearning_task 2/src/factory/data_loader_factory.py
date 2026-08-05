"""
Factory class for creating data loaders.
"""

from src.data.image_data_loader import ImageDataLoader
from src.data.tabular_data_loader import TabularDataLoader
from src.data.text_data_loader import TextDataLoader
from src.data.transfer_data_loader import TransferDataLoader

class DataLoaderFactory:
    @staticmethod
    def create_data_loader(config):
        model_name = config["model_name"].upper()
        loaders = {
            "ANN": TabularDataLoader,
            "CNN": ImageDataLoader,
            "RNN": TextDataLoader,
            "LSTM": TextDataLoader,
            "TRANSFER": TransferDataLoader,
}
        if model_name not in loaders:
            raise ValueError(
                f"Unsupported model: {model_name}")
        return loaders[model_name](config)