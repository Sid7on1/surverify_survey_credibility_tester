import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
ACS_INCOME_URL = 'https://www.census.gov/data/datasets/time-series/demo/acs/acs5.html'
SYNTHETIC_DATA_SIZE = 1000
SYNTHETIC_FEATURES = 10
SYNTHETIC_NOISE = 0.1

# Define data structures
@dataclass
class SyntheticData:
    X: np.ndarray
    y: np.ndarray

@dataclass
class ACSIncomeData:
    data: pd.DataFrame

# Define exception classes
class DataGenerationError(Exception):
    pass

class ACSIncomeLoadError(Exception):
    pass

# Define configuration class
class DataGeneratorConfig:
    def __init__(self, data_size: int = SYNTHETIC_DATA_SIZE, features: int = SYNTHETIC_FEATURES, noise: float = SYNTHETIC_NOISE):
        self.data_size = data_size
        self.features = features
        self.noise = noise

# Define main class
class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.lock = Lock()

    def generate_synthetic_data(self) -> SyntheticData:
        """
        Generate synthetic regression data.

        Returns:
            SyntheticData: A dataclass containing the feature matrix X and target vector y.
        """
        with self.lock:
            try:
                X, y = make_regression(n_samples=self.config.data_size, n_features=self.config.features, noise=self.config.noise)
                return SyntheticData(X, y)
            except Exception as e:
                logger.error(f'Error generating synthetic data: {str(e)}')
                raise DataGenerationError('Failed to generate synthetic data')

    def load_acs_income(self) -> ACSIncomeData:
        """
        Load ACS income data from the US Census Bureau.

        Returns:
            ACSIncomeData: A dataclass containing the loaded ACS income data.
        """
        with self.lock:
            try:
                data = pd.read_csv(ACS_INCOME_URL)
                return ACSIncomeData(data)
            except Exception as e:
                logger.error(f'Error loading ACS income data: {str(e)}')
                raise ACSIncomeLoadError('Failed to load ACS income data')

    def create_distribution_shift(self, data: SyntheticData, shift_type: str = 'mean') -> SyntheticData:
        """
        Create a distribution shift in the synthetic data.

        Args:
            data (SyntheticData): The synthetic data to shift.
            shift_type (str, optional): The type of shift to apply. Defaults to 'mean'.

        Returns:
            SyntheticData: The shifted synthetic data.
        """
        with self.lock:
            try:
                if shift_type == 'mean':
                    # Apply mean shift
                    data.X += np.random.normal(0, 1, size=data.X.shape)
                elif shift_type == 'variance':
                    # Apply variance shift
                    data.X *= np.random.normal(1, 0.1, size=data.X.shape)
                else:
                    logger.warning(f'Unknown shift type: {shift_type}')
                return data
            except Exception as e:
                logger.error(f'Error creating distribution shift: {str(e)}')
                raise DataGenerationError('Failed to create distribution shift')

# Define utility functions
def split_data(data: SyntheticData, test_size: float = 0.2) -> Tuple[SyntheticData, SyntheticData]:
    """
    Split the synthetic data into training and testing sets.

    Args:
        data (SyntheticData): The synthetic data to split.
        test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.

    Returns:
        Tuple[SyntheticData, SyntheticData]: The training and testing data.
    """
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=test_size)
    return SyntheticData(X_train, y_train), SyntheticData(X_test, y_test)

def scale_data(data: SyntheticData) -> SyntheticData:
    """
    Scale the synthetic data using StandardScaler.

    Args:
        data (SyntheticData): The synthetic data to scale.

    Returns:
        SyntheticData: The scaled synthetic data.
    """
    scaler = StandardScaler()
    data.X = scaler.fit_transform(data.X)
    return data

# Define main function
def main():
    config = DataGeneratorConfig()
    generator = DataGenerator(config)
    data = generator.generate_synthetic_data()
    logger.info(f'Generated synthetic data with shape: {data.X.shape}')
    shifted_data = generator.create_distribution_shift(data)
    logger.info(f'Created distribution shift in synthetic data')
    train_data, test_data = split_data(shifted_data)
    logger.info(f'Split data into training and testing sets')
    scaled_train_data = scale_data(train_data)
    logger.info(f'Scaled training data')

if __name__ == '__main__':
    main()