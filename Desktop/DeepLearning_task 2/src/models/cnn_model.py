"""
Convolutional Neural Network (CNN) model.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense,Input,)
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel
class CNNModel(BaseModel):
    """
    CNN model for image classification.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
    def build_model(self):
        """
        Build and compile the CNN model.
        """
        model = Sequential(
            [
                Input(shape=tuple(self.config["input_shape"])),

                Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(
                    self.config["num_classes"],
                    activation="softmax",),])
        model.compile(
            optimizer=Adam(
                learning_rate=self.config["learning_rate"]),
            loss=self.config["loss"],
            metrics=self.config["metrics"],)
        self.model = model
        return self.model