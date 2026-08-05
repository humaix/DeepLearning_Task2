"""
Base evaluator for machine learning models.
"""

from abc import ABC, abstractmethod
class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """
    @abstractmethod
    def evaluate(self, model, x_test, y_test):
        """
        Evaluate the trained model.
        """
        pass