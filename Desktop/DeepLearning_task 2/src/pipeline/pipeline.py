"""
Pipeline for training and evaluating deep learning models.
"""

from src.factory.data_loader_factory import DataLoaderFactory
from src.factory.evaluator_factory import EvaluatorFactory
from src.factory.model_factory import ModelFactory
from src.trainer.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualizer import Visualizer

class Pipeline:
    """
    Executes the complete deep learning workflow.
    """
    def __init__(self, config):
        self.config = config
        self.model_name = config["model_name"].upper()
        self.logger = get_logger(__name__)
    def run(self):
        self.logger.info("Pipeline started.")

        # Load Data
        data_loader = DataLoaderFactory.create_data_loader(self.config)
        data = data_loader.load_data()

        # Build Model
        model_builder = ModelFactory.create_model(self.config)
        model = model_builder.build_model()
        trainer = Trainer()

        # ANN
        if self.model_name == "ANN":
            x_train, x_test, y_train, y_test = data
            history = trainer.train(
                model=model,
                x_train=x_train,
                y_train=y_train,
                config=self.config)
            Visualizer().plot_history(history)
            evaluator = EvaluatorFactory.get_evaluator(
                self.model_name)
            results = evaluator.evaluate(
                model=model,
                x_test=x_test,
                y_test=y_test)

        # TRANSFER LEARNING
        elif self.model_name == "TRANSFER":
            train_dataset, test_dataset = data
            history = trainer.train(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=self.config)
            Visualizer().plot_history(history)
            evaluator = EvaluatorFactory.get_evaluator(
                self.model_name)
            results = evaluator.evaluate(
                model=model,
                test_dataset=test_dataset )
        # CNN / RNN / LSTM
        else:
            x_train, x_test, y_train, y_test = data
            history = trainer.train(
                model=model,
                x_train=x_train,
                y_train=y_train,
                config=self.config)
            Visualizer().plot_history(history)
            evaluator = EvaluatorFactory.get_evaluator(
                self.model_name)
            if self.model_name == "CNN":
                results = evaluator.evaluate(
                    model=model,
                    x_test=x_test,
                    y_test=y_test)
            else:
                results = evaluator.evaluate(
                    model=model,
                    x_test=x_test,
                    y_test=y_test)
        self.logger.info("Pipeline completed successfully.")
        return results