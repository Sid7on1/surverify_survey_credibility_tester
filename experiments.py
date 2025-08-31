import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from surverify import SurVerify
from regression_models import LinearRegression, RidgeRegression
from data_generator import generate_synthetic_data, load_acs_income_data
import numpy as np
import logging
import argparse
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_synthetic_experiments(n_samples: List[int], dim: int, n_trials: int, eta: float,
                              delta: float, l2_penalty: float = 0.0) -> pd.DataFrame:
    """
    Run synthetic experiments to evaluate SurVerify's performance.

    Parameters:
    n_samples (List[int]): List of sample sizes to test.
    dim (int): Dimensionality of the data.
    n_trials (int): Number of trials to run for each sample size.
    eta (float): Velocity threshold parameter for SurVerify.
    delta (float): Failure probability for SurVerify.
    l2_penalty (float, optional): L2 penalty for ridge regression. Defaults to 0.0.

    Returns:
    pd.DataFrame: DataFrame containing the results, including sample size, trial number,
                 acceptance rate, and sample complexity.
    """
    results = []
    for n in n_samples:
        acceptance_rates = []
        for trial in range(n_trials):
            logger.info(f"Running trial {trial + 1}/{n_trials} for n = {n}")
            # Generate synthetic data
            X_train, y_train, X_test, y_test = generate_synthetic_data(n, dim)
            # Fit regression models
            lr = LinearRegression().fit(X_train, y_train)
            rr = RidgeRegression(l2_penalty).fit(X_train, y_train)
            # Initialize SurVerify
            sv = SurVerify(X_train, y_train, X_test, y_test, lr, rr, eta, delta)
            # Run SurVerify and get sample complexity
            sv.run()
            acceptance_rates.append(sv.acceptance_rate)
        results.append({'n': n, 'trial': range(n_trials), 'acceptance_rate': acceptance_rates,
                        'sample_complexity': sv.sample_complexity})
    return pd.DataFrame(results)

def run_acs_experiments(n_samples: List[int], n_trials: int, eta: float, delta: float,
                       l2_penalty: float = 0.0) -> pd.DataFrame:
    """
    Run experiments using ACS Income data to evaluate SurVerify's performance.

    Parameters:
    n_samples (List[int]): List of sample sizes to test.
    n_trials (int): Number of trials to run for each sample size.
    eta (float): Velocity threshold parameter for SurVerify.
    delta (float): Failure probability for SurVerify.
    l2_penalty (float, optional): L2 penalty for ridge regression. Defaults to 0.0.

    Returns:
    pd.DataFrame: DataFrame containing the results, including sample size, trial number,
                 acceptance rate, and sample complexity.
    """
    df = load_acs_income_data()
    X = df.drop('income', axis=1)
    y = df['income']
    results = []
    for n in n_samples:
        acceptance_rates = []
        for trial in range(n_trials):
            logger.info(f"Running trial {trial + 1}/{n_trials} for n = {n}")
            # Sample data points randomly
            idx = np.random.choice(X.shape[0], n, replace=True)
            X_train, y_train = X.iloc[idx], y.iloc[idx]
            X_test, y_test = X.drop(idx), y.drop(idx)
            # Fit regression models
            lr = LinearRegression().fit(X_train, y_train)
            rr = RidgeRegression(l2_penalty).fit(X_train, y_train)
            # Initialize SurVerify
            sv = SurVerify(X_train, y_train, X_test, y_test, lr, rr, eta, delta)
            # Run SurVerify and get sample complexity
            sv.run()
            acceptance_rates.append(sv.acceptance_rate)
        results.append({'n': n, 'trial': range(n_trials), 'acceptance_rate': acceptance_rates,
                        'sample_complexity': sv.sample_complexity})
    return pd.DataFrame(results)

def plot_acceptance_rates(results: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the acceptance rates from the experimental results.

    Parameters:
    results (pd.DataFrame): DataFrame containing experimental results.
    save_path (str, optional): Path to save the plot. If None, show the plot.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    for n, group in results.groupby('n'):
        ax.plot(group['trial'], group['acceptance_rate'].mean(axis=0), label=f'n = {n}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rates vs Trials')
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_sample_complexity(results: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot the sample complexity from the experimental results.

    Parameters:
    results (pd.DataFrame): DataFrame containing experimental results.
    save_path (str, optional): Path to save the plot. If None, show the plot.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    for n, group in results.groupby('n'):
        ax.scatter(n, group['sample_complexity'].mean())
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Sample Complexity')
    ax.set_title('Sample Complexity vs Sample Size')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', nargs='+', type=int, required=True,
                        help='List of sample sizes to test')
    parser.add_argument('--dim', type=int, default=10, help='Dimensionality of synthetic data')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Velocity threshold parameter for SurVerify')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='Failure probability for SurVerify')
    parser.add_argument('--l2_penalty', type=float, default=0.0,
                        help='L2 penalty for ridge regression')
    parser.add_argument('--acs_data_path', type=str, default='acs_income_data.csv',
                        help='Path to ACS Income data CSV file')
    parser.add_argument('--results_path', type=str, default='results.csv',
                        help='Path to save experimental results')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    # Run synthetic experiments
    synthetic_results = run_synthetic_experiments(args.n_samples, args.dim, args.n_trials,
                                                args.eta, args.delta, args.l2_penalty)
    synthetic_results.to_csv(args.results_path)
    plot_acceptance_rates(synthetic_results, os.path.join(args.plots_dir, 'synthetic_acceptance.png'))
    plot_sample_complexity(synthetic_results, os.path.join(args.plots_dir, 'synthetic_complexity.png'))

    # Run ACS experiments
    acs_results = run_acs_experiments(args.n_samples, args.n_trials, args.eta, args.delta,
                                     args.l2_penalty)
    acs_results.to_csv(args.results_path)
    plot_acceptance_rates(acs_results, os.path.join(args.plots_dir, 'acs_acceptance.png'))
    plot_sample_complexity(acs_results, os.path.join(args.plots_dir, 'acs_complexity.png'))