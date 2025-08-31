import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.kernel_approximation import RBFSampler
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple
import logging
import logging.config
import yaml
import os

# Load logging configuration
logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
with open(logging_config_path, 'r') as f:
    logging_config = yaml.safe_load(f.read())
logging.config.dictConfig(logging_config)

# Get logger
logger = logging.getLogger(__name__)

class LassoModel(BaseEstimator, RegressorMixin):
    """
    Lasso regression model with bounded constraints.

    Parameters:
    - alpha (float): regularization strength
    - max_iter (int): maximum number of iterations
    - tol (float): tolerance for convergence
    - fit_intercept (bool): whether to fit the intercept
    - normalize (bool): whether to normalize the data
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4, fit_intercept: bool = True, normalize: bool = False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoModel':
        """
        Fit the model to the data.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - self
        """
        self.lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept, normalize=self.normalize)
        self.lasso.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given feature matrix.

        Parameters:
        - X (np.ndarray): feature matrix

        Returns:
        - np.ndarray: predicted target values
        """
        return self.lasso.predict(X)

    def get_rademacher_bound(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Rademacher complexity of the model.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - float: Rademacher complexity
        """
        # Compute the Rademacher complexity using the formula from the paper
        # This is a simplified implementation and may not be exact
        n_samples, n_features = X.shape
        rademacher_bound = np.mean(np.abs(self.lasso.coef_)) / np.sqrt(n_samples)
        return rademacher_bound


class RidgeModel(BaseEstimator, RegressorMixin):
    """
    Ridge regression model with bounded constraints.

    Parameters:
    - alpha (float): regularization strength
    - max_iter (int): maximum number of iterations
    - tol (float): tolerance for convergence
    - fit_intercept (bool): whether to fit the intercept
    - normalize (bool): whether to normalize the data
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4, fit_intercept: bool = True, normalize: bool = False):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeModel':
        """
        Fit the model to the data.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - self
        """
        self.ridge = Ridge(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept, normalize=self.normalize)
        self.ridge.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given feature matrix.

        Parameters:
        - X (np.ndarray): feature matrix

        Returns:
        - np.ndarray: predicted target values
        """
        return self.ridge.predict(X)

    def get_rademacher_bound(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Rademacher complexity of the model.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - float: Rademacher complexity
        """
        # Compute the Rademacher complexity using the formula from the paper
        # This is a simplified implementation and may not be exact
        n_samples, n_features = X.shape
        rademacher_bound = np.mean(np.abs(self.ridge.coef_)) / np.sqrt(n_samples)
        return rademacher_bound


class KernelModel(BaseEstimator, RegressorMixin):
    """
    Kernel regression model with bounded constraints.

    Parameters:
    - kernel (str): kernel type (e.g. 'rbf', 'linear')
    - gamma (float): kernel parameter
    - degree (int): kernel parameter
    - coef0 (float): kernel parameter
    """

    def __init__(self, kernel: str = 'rbf', gamma: float = 1.0, degree: int = 3, coef0: float = 1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelModel':
        """
        Fit the model to the data.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - self
        """
        self.sampler = RBFSampler(gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        self.sampler.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given feature matrix.

        Parameters:
        - X (np.ndarray): feature matrix

        Returns:
        - np.ndarray: predicted target values
        """
        return self.sampler.transform(X)

    def get_rademacher_bound(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Rademacher complexity of the model.

        Parameters:
        - X (np.ndarray): feature matrix
        - y (np.ndarray): target vector

        Returns:
        - float: Rademacher complexity
        """
        # Compute the Rademacher complexity using the formula from the paper
        # This is a simplified implementation and may not be exact
        n_samples, n_features = X.shape
        rademacher_bound = np.mean(np.abs(self.sampler.weights_)) / np.sqrt(n_samples)
        return rademacher_bound