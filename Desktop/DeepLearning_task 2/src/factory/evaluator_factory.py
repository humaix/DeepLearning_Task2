



"""
Factory for selecting the correct evaluator.
"""
from src.evaluation.binary_evaluator import BinaryEvaluator
from src.evaluation.multiclass_evaluator import MultiClassEvaluator
class EvaluatorFactory:
    @staticmethod
    def get_evaluator(model_name: str):
        model_name = model_name.upper()
        if model_name in ["ANN", "RNN", "LSTM", "TRANSFER"]:
            return BinaryEvaluator()
        elif model_name == "CNN":
            return MultiClassEvaluator()
        else:
            raise ValueError(f"Unsupported model: {model_name}")