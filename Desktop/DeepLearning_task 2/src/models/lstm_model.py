"""
Long Short-Term Memory (LSTM) model.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Embedding,LSTM,Dense,)
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel
class LSTMModel(BaseModel):
    """
    LSTM model for text classification.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        model = Sequential(
            [
                Embedding(
                    input_dim=self.config["vocabulary_size"],
                    output_dim=self.config["embedding_dim"],
                ),
                LSTM(
                    units=self.config["lstm_units"]
                ),
                Dense(
                    units=1,
                    activation="sigmoid")])
        model.compile(
            optimizer=Adam(
                learning_rate=self.config["learning_rate"]
            ),
            loss=self.config["loss"],
            metrics=self.config["metrics"]
        )
        self.model = model
        return self.model