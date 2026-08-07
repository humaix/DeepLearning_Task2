"""
Data Loader for Transfer Learning.
"""

import logging
import tensorflow as tf
logger = logging.getLogger(__name__)
class TransferDataLoader:
    def __init__(self, config):
        self.config = config
        self.image_size = tuple(config["input_shape"][:2])
        self.batch_size = config["batch_size"]
        self.train_path = "data/train"
        self.test_path = "data/test"

    def load_data(self):
        logger.info("Loading Cats vs Dogs dataset.")
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_path,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True)

        test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_path,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False)

        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(AUTOTUNE)
        test_dataset = test_dataset.prefetch(AUTOTUNE)

        return train_dataset, test_dataset