"""
Generic Trainer class for all deep learning models.
"""

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from src.utils.logger import get_logger
class Trainer:
    """
    Handles training for all models.
    """
    def __init__(self):
        self.logger = get_logger(__name__)
    def train(
        self,
        model,
        config,
        x_train=None,
        y_train=None,
        train_dataset=None,
        test_dataset=None,
    ):
        """
        Train any deep learning model.
        """
        self.logger.info("Training started.")
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,),
            ModelCheckpoint(
                filepath=f"models/{config['model_name'].lower()}_best.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,),]

        # Transfer Learning (tf.data.Dataset)
        if train_dataset is not None:
            history = model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=config["epochs"],
                callbacks=callbacks,
                verbose=1,)

        # ANN / CNN / RNN / LSTM
        else:
            history = model.fit(
                x_train,
                y_train,
                validation_split=config["validation_split"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                callbacks=callbacks,
                verbose=1,)
        self.logger.info("Training completed.")
        return history