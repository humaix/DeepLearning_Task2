
"""
Utility functions for loading YAML configuration files.
"""
from pathlib import Path

import yaml
class ConfigLoader:
    """
    Loads configuration settings from a YAML file.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
    def load(self) -> dict:
        """
        Read the YAML file and return its contents.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config