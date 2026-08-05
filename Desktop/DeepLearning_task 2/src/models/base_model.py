"""
Base class for all deep learning models.
"""
from abc import ABC, abstractmethod
from tensorflow.keras import Model
class BaseModel(ABC):
    """
    Abstract base class for all deep learning models.
    """
    def __init__(self):
        self.model: Model | None = None
    @abstractmethod
    def build_model(self) -> Model:
        """
        Build the model architecture.
        """
        pass
    def get_model(self) -> Model:
        """
        Return the model instance.
        """
        return self.model