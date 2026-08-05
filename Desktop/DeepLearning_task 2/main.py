"""
Main entry point of the Deep Learning project.
"""
from src.pipeline.pipeline import Pipeline
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

def main():
    logger = get_logger(__name__)
    logger.info("Project started.")
    config = ConfigLoader(
        "configs/transfer_config.yaml"
    ).load()
    pipeline = Pipeline(config)
    results = pipeline.run()
    logger.info(
        f"Final Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()