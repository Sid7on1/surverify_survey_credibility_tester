import numpy as np
import logging
from typing import Tuple, Dict, List
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import erf
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FDDMetric:
    """
    Implementation of Functional Distance of Distributions (FDD) metric calculation.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FDDMetric object.

        Args:
        config (Dict): Configuration dictionary.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_assumptions(self, x: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate the assumptions of the FDD metric.

        Args:
        x (np.ndarray): First distribution.
        y (np.ndarray): Second distribution.

        Returns:
        bool: True if assumptions are valid, False otherwise.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            self.logger.error("Both inputs must be numpy arrays.")
            return False

        if x.shape != y.shape:
            self.logger.error("Both distributions must have the same shape.")
            return False

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            self.logger.error("Both distributions must not contain NaN values.")
            return False

        return True

    def fdd_decomposition(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Decompose the FDD metric into its components.

        Args:
        x (np.ndarray): First distribution.
        y (np.ndarray): Second distribution.

        Returns:
        Tuple[float, float]: Decomposed FDD metric components.
        """
        if not self.validate_assumptions(x, y):
            return None, None

        # Compute the mean and standard deviation of the distributions
        mean_x = np.mean(x)
        std_x = np.std(x)
        mean_y = np.mean(y)
        std_y = np.std(y)

        # Compute the velocity-threshold
        velocity_threshold = self.config['velocity_threshold']

        # Compute the Flow Theory component
        flow_theory_component = self.flow_theory_component(x, y, mean_x, mean_y, std_x, std_y, velocity_threshold)

        # Compute the velocity component
        velocity_component = self.velocity_component(x, y, mean_x, mean_y, std_x, std_y, velocity_threshold)

        return flow_theory_component, velocity_component

    def flow_theory_component(self, x: np.ndarray, y: np.ndarray, mean_x: float, mean_y: float, std_x: float, std_y: float, velocity_threshold: float) -> float:
        """
        Compute the Flow Theory component of the FDD metric.

        Args:
        x (np.ndarray): First distribution.
        y (np.ndarray): Second distribution.
        mean_x (float): Mean of the first distribution.
        mean_y (float): Mean of the second distribution.
        std_x (float): Standard deviation of the first distribution.
        std_y (float): Standard deviation of the second distribution.
        velocity_threshold (float): Velocity-threshold.

        Returns:
        float: Flow Theory component of the FDD metric.
        """
        # Compute the difference between the means
        mean_diff = mean_x - mean_y

        # Compute the integral of the difference between the cumulative distributions
        integral, _ = quad(self.integral_flow_theory, -np.inf, np.inf, args=(x, y, mean_diff, std_x, std_y, velocity_threshold))

        return integral

    def velocity_component(self, x: np.ndarray, y: np.ndarray, mean_x: float, mean_y: float, std_x: float, std_y: float, velocity_threshold: float) -> float:
        """
        Compute the velocity component of the FDD metric.

        Args:
        x (np.ndarray): First distribution.
        y (np.ndarray): Second distribution.
        mean_x (float): Mean of the first distribution.
        mean_y (float): Mean of the second distribution.
        std_x (float): Standard deviation of the first distribution.
        std_y (float): Standard deviation of the second distribution.
        velocity_threshold (float): Velocity-threshold.

        Returns:
        float: Velocity component of the FDD metric.
        """
        # Compute the difference between the means
        mean_diff = mean_x - mean_y

        # Compute the integral of the difference between the cumulative distributions
        integral, _ = quad(self.integral_velocity, -np.inf, np.inf, args=(x, y, mean_diff, std_x, std_y, velocity_threshold))

        return integral

    def integral_flow_theory(self, t: float, args) -> float:
        """
        Compute the integral of the difference between the cumulative distributions for the Flow Theory component.

        Args:
        t (float): Integration variable.
        args: Additional arguments.

        Returns:
        float: Integral value.
        """
        x, y, mean_diff, std_x, std_y, velocity_threshold = args

        # Compute the cumulative distributions
        cdf_x = norm.cdf(t, loc=mean_x, scale=std_x)
        cdf_y = norm.cdf(t, loc=mean_y, scale=std_y)

        # Compute the difference between the cumulative distributions
        diff = cdf_x - cdf_y

        # Compute the velocity-threshold
        velocity = np.abs(t - mean_x) / std_x

        # Return the integral value
        return diff * np.exp(-velocity ** 2 / (2 * velocity_threshold ** 2))

    def integral_velocity(self, t: float, args) -> float:
        """
        Compute the integral of the difference between the cumulative distributions for the velocity component.

        Args:
        t (float): Integration variable.
        args: Additional arguments.

        Returns:
        float: Integral value.
        """
        x, y, mean_diff, std_x, std_y, velocity_threshold = args

        # Compute the cumulative distributions
        cdf_x = norm.cdf(t, loc=mean_x, scale=std_x)
        cdf_y = norm.cdf(t, loc=mean_y, scale=std_y)

        # Compute the difference between the cumulative distributions
        diff = cdf_x - cdf_y

        # Compute the velocity-threshold
        velocity = np.abs(t - mean_x) / std_x

        # Return the integral value
        return diff * np.exp(-velocity ** 2 / (2 * velocity_threshold ** 2))

    def compute_fdd(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Functional Distance of Distributions (FDD) metric.

        Args:
        x (np.ndarray): First distribution.
        y (np.ndarray): Second distribution.

        Returns:
        float: FDD metric value.
        """
        if not self.validate_assumptions(x, y):
            return None

        # Decompose the FDD metric into its components
        flow_theory_component, velocity_component = self.fdd_decomposition(x, y)

        # Return the FDD metric value
        return flow_theory_component + velocity_component

def main():
    # Set up the configuration
    config = {
        'velocity_threshold': 1.0
    }

    # Create an instance of the FDDMetric class
    fdd_metric = FDDMetric(config)

    # Generate some random data
    np.random.seed(0)
    x = np.random.normal(loc=0, scale=1, size=1000)
    y = np.random.normal(loc=1, scale=1, size=1000)

    # Compute the FDD metric
    fdd_value = fdd_metric.compute_fdd(x, y)

    # Print the FDD metric value
    print(f"FDD metric value: {fdd_value}")

if __name__ == "__main__":
    main()