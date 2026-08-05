"""
Recurrent Neural Network (RNN) model.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Embedding,SimpleRNN,Dense,)
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel
class RNNModel(BaseModel):
    """
    Simple RNN model for text classification.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
    def build_model(self):
        """
        Build and compile the RNN model.
        """
        model = Sequential(
            [
                Embedding(
                    input_dim=self.config["vocabulary_size"],
                    output_dim=self.config["embedding_dim"],
                    input_length=self.config["max_sequence_length"],
                ),
                SimpleRNN(
                    units=self.config["rnn_units"]),
                Dense(
                    1,
                    activation="sigmoid",),])
        model.compile(
            optimizer=Adam(
                learning_rate=self.config["learning_rate"]),
            loss=self.config["loss"],
            metrics=self.config["metrics"],)
        self.model = model
        return self.model