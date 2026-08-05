"""
Transfer Learning Model using MobileNetV2.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense,Dropout,GlobalAveragePooling2D,)
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel
class TransferModel(BaseModel):
    """
    Transfer Learning model using MobileNetV2.
    """

    def __init__(self, config):

        super().__init__()

        self.config = config
    def build_model(self):


        # Pretrained Base Model
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=tuple(
                self.config["input_shape"]))

        # Freeze pretrained layers
        base_model.trainable = False
        # Input Layer
        inputs = tf.keras.Input(
            shape=tuple(
                self.config["input_shape"]))
        # MobileNetV2 preprocessing
        x = tf.keras.applications.mobilenet_v2.preprocess_input(
            inputs)
        # Base model
        x = base_model(
            x,
            training=False)
        # Classification Head
        x = GlobalAveragePooling2D()(x)
        x = Dense(
            128,
            activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(
            1,
            activation="sigmoid")(x)

        # Final Model
        self.model = Model(
            inputs,
            outputs
        )
        # Compile
        self.model.compile(
            optimizer=Adam(
                learning_rate=self.config[
                    "learning_rate"
                ]
            ),
            loss=self.config["loss"],
            metrics=self.config["metrics"]
        )
        return self.model