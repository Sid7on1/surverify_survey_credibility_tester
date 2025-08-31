import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 5
    velocity_threshold: float = 0.01
    flow_theory_parameter: float = 0.5
    distribution_distance_metric: str = "fdd"  # Functional Distance of Distributions
    # ... other hyperparameters and settings ...

    def __post_init__(self):
        self.validate_parameters()

    def validate_parameters(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if self.num_epochs < 0:
            raise ValueError("Number of epochs must be a non-negative integer.")
        # ... validate other parameters ...

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

def load_config(config_file: str) -> Config:
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    return Config.from_dict(config_dict)

def save_config(config: Config, config_file: str) -> None:
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=4)

def validate_parameters(config: Config) -> None:
    config.validate_parameters()

# Example usage
if __name__ == "__main__":
    config_file = "config.json"

    # Load config from file
    if os.path.exists(config_file):
        config = load_config(config_file)
        logger.info("Config loaded from file:")
    else:
        config = Config()
        logger.info("Default config used:")

    # Print config settings
    logger.info(json.dumps(config.to_dict(), indent=4))

    # Validate parameters
    validate_parameters(config)
    logger.info("Parameters validated.")

    # Save config to file
    save_config(config, config_file)
    logger.info("Config saved to file.")