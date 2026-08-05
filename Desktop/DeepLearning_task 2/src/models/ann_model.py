"""
Artificial Neural Network (ANN) model.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel
class ANNModel(BaseModel):
    """
    Artificial Neural Network for binary classification.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
    def build_model(self):
        """
        Build and compile the ANN model.
        """
        model = Sequential([Input(shape=(30,)),
                            Dense(64, activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(1, activation="sigmoid")])
        model.compile(optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss=self.config["loss"],
            metrics=self.config["metrics"])
        self.model = model

        return self.model