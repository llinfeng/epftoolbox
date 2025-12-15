"""
Forecast Pipeline for Electricity Price Forecasting

This module provides a flexible pipeline for running LEAR and DNN forecasts
on electricity price data. It supports both epftoolbox data loading and
local cached data, with full control over hyperparameters and model configurations.

The pipeline is designed to be future-proof and extensible for other forecasting
libraries beyond epftoolbox.
"""

import os
import warnings
import time
import pickle as pc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import importlib

from epftoolbox.data import read_data, scaling
from epftoolbox.models._lear import LEAR
import epftoolbox.models._lear as _lear
from epftoolbox.models import hyperparameter_optimizer, evaluate_dnn_in_test_dataset
from epftoolbox.evaluation import MAE, RMSE, MAPE, sMAPE, MASE, rMAE
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data_from_csv(
    csv_path: Union[str, Path] = 'datasets/hourly_data_all_markets.csv',
    market: str = 'PJM'
) -> pd.DataFrame:
    """
    Load data from local CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file containing hourly data
    market : str
        Market name to filter (default: 'PJM')
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe for the specified market
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df[df['mkt'] == market]
    return df


def load_data_from_epftoolbox(
    dataset: str = 'PJM',
    years_test: int = 2,
    path: Union[str, Path] = 'datasets'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data using epftoolbox's read_data function.
    
    Parameters:
    -----------
    dataset : str
        Dataset name (default: 'PJM')
    years_test : int
        Number of years for test set (default: 2)
    path : str or Path
        Path to datasets folder
    
    Returns:
    --------
    tuple of pd.DataFrame
        (df_train, df_test) tuple
    """
    return read_data(dataset=dataset, years_test=years_test, path=str(path))


def get_actual_values(
    df: pd.DataFrame,
    target_date: Union[str, pd.Timestamp]
) -> np.ndarray:
    """
    Extract actual price values for a target date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'DateTime' and 'Price' columns
    target_date : str or pd.Timestamp
        Target date to extract values for
    
    Returns:
    --------
    np.ndarray
        Array of 24 hourly price values
    """
    target_date = pd.to_datetime(target_date)
    actual = df[
        pd.to_datetime(df['DateTime']).dt.date == target_date.date()
    ]['Price'].values
    return actual


# ============================================================================
# LEAR Model Functions
# ============================================================================

class LEARFixedAlpha(LEAR):
    """
    LEAR variant that uses a fixed alpha instead of LARS/AIC.
    Useful when n_samples < n_features (e.g., with small calibration windows).
    """
    def __init__(self, calibration_window: int, alpha_fixed: float):
        super().__init__(calibration_window=calibration_window)
        self._alpha_fixed = alpha_fixed

    def recalibrate(self, Xtrain, Ytrain):
        """Recalibrate using fixed alpha instead of LassoLarsIC."""
        [Ytrain], self.scalerY = scaling([Ytrain], 'Invariant')
        [Xtrain_no_dummies], self.scalerX = scaling([Xtrain[:, :-7]], 'Invariant')
        Xtrain[:, :-7] = Xtrain_no_dummies
        self.models = {}
        for h in range(24):
            model = Lasso(max_iter=2500, alpha=self._alpha_fixed)
            model.fit(Xtrain, Ytrain[:, h])
            self.models[h] = model


def run_lear_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_date: Union[str, pd.Timestamp],
    calibration_window: int,
    alpha_fixed: Optional[float] = None,
    reload_lear: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Run LEAR forecast for a single day.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Test data
    target_date : str or pd.Timestamp
        Target date for forecast
    calibration_window : int
        Calibration window in days
    alpha_fixed : float, optional
        Fixed alpha value for Lasso (if None, uses AIC/LARS)
    reload_lear : bool
        Whether to reload LEAR module (default: True)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    tuple
        (prediction array, metadata dict)
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running LEAR forecast - Calibration Window: {calibration_window} days")
        print(f"Target Date: {pd.to_datetime(target_date).date()}")
        print(f"{'='*60}")
    
    if reload_lear:
        importlib.reload(_lear)
    
    target_date = pd.to_datetime(target_date)
    
    # Prepare data: available up to target day, hide target day prices
    data_available = pd.concat([
        df_train,
        df_test.loc[:target_date + pd.Timedelta(hours=23)]
    ])
    data_available.loc[
        target_date:target_date + pd.Timedelta(hours=23), 'Price'
    ] = np.nan
    
    # Try AIC/LARS first, fall back to fixed alpha if needed
    metadata = {
        'calibration_window': calibration_window,
        'target_date': target_date,
        'mode': None,
        'alpha': None
    }
    
    if alpha_fixed is not None:
        if verbose:
            print(f"Using fixed alpha: {alpha_fixed}")
        # Use fixed alpha directly
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = LEARFixedAlpha(
            calibration_window=calibration_window,
            alpha_fixed=alpha_fixed
        )
        metadata['mode'] = 'fixed_alpha'
        metadata['alpha'] = alpha_fixed
        
        prediction = model.recalibrate_and_forecast_next_day(
            df=data_available,
            next_day_date=target_date,
            calibration_window=calibration_window,
        )
        metadata['alphas'] = [alpha_fixed] * 24
    else:
        # Try AIC/LARS first - wrap the entire forecast call in try-except
        # because the error occurs during recalibrate_and_forecast_next_day
        try:
            if verbose:
                print("Attempting AIC/LARS mode...")
            model = LEAR(calibration_window=calibration_window)
            prediction = model.recalibrate_and_forecast_next_day(
                df=data_available,
                next_day_date=target_date,
                calibration_window=calibration_window,
            )
            metadata['mode'] = 'AIC'
            if verbose:
                print("✓ AIC/LARS mode successful")
            # Extract alphas if available
            try:
                metadata['alphas'] = [model.models[h].alpha for h in range(24)]
            except:
                metadata['alphas'] = None
        except ValueError as e:
            # Fall back to fixed alpha if AIC fails (n_samples < n_features)
            if verbose:
                print(f"⚠ AIC/LARS failed (n_samples < n_features), falling back to fixed alpha")
            fixed_alpha = 1e-2  # slightly stronger shrinkage to help convergence
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = LEARFixedAlpha(
                calibration_window=calibration_window,
                alpha_fixed=fixed_alpha
            )
            prediction = model.recalibrate_and_forecast_next_day(
                df=data_available,
                next_day_date=target_date,
                calibration_window=calibration_window,
            )
            metadata['mode'] = 'fixed_alpha'
            metadata['alpha'] = fixed_alpha
            metadata['alphas'] = [fixed_alpha] * 24
    
    elapsed_time = time.time() - start_time
    metadata['computation_time'] = elapsed_time
    
    if verbose:
        print(f"✓ Completed in {elapsed_time:.2f} seconds")
        print(f"  Mode: {metadata['mode']}, Alpha: {metadata.get('alpha', 'varies')}")
        print(f"{'='*60}\n")
    
    return prediction[0], metadata


def run_lear_forecast_multiday(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_start_date: Union[str, pd.Timestamp],
    n_days: int,
    calibration_windows: List[int],
    alpha_fixed: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Tuple[np.ndarray, Dict]]]:
    """
    Run LEAR forecasts for multiple consecutive days with daily recalibration.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Test data
    target_start_date : str or pd.Timestamp
        Start date for forecasts
    n_days : int
        Number of consecutive days to forecast
    calibration_windows : list of int
        List of calibration windows to test
    alpha_fixed : float, optional
        Fixed alpha value (if None, uses AIC/LARS with fallback)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    dict
        Nested dictionary: {date_str: {model_name: (prediction, metadata)}}
    """
    target_start_date = pd.to_datetime(target_start_date)
    all_results = {}
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"LEAR Multi-Day Forecast Processing")
        print(f"Start Date: {target_start_date.date()}")
        print(f"Number of Days: {n_days}")
        print(f"Calibration Windows: {calibration_windows}")
        print(f"{'#'*70}\n")
    
    for day_idx in range(n_days):
        current_date = target_start_date + pd.Timedelta(days=day_idx)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Day {day_idx + 1}/{n_days}: {current_date.date()}")
            print(f"{'='*70}")
        
        day_results = run_lear_forecasts(
            df_train=df_train,
            df_test=df_test,
            target_date=current_date,
            calibration_windows=calibration_windows,
            alpha_fixed=alpha_fixed,
            verbose=verbose
        )
        
        all_results[current_date.strftime('%Y-%m-%d')] = day_results
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"Multi-Day LEAR Forecast Complete")
        print(f"Total days processed: {n_days}")
        print(f"{'#'*70}\n")
    
    return all_results


def run_lear_forecasts(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_date: Union[str, pd.Timestamp],
    calibration_windows: List[int],
    alpha_fixed: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Run multiple LEAR forecasts with different calibration windows.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data
    df_test : pd.DataFrame
        Test data
    target_date : str or pd.Timestamp
        Target date for forecast
    calibration_windows : list of int
        List of calibration windows to test
    alpha_fixed : float, optional
        Fixed alpha value (if None, uses AIC/LARS with fallback)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    dict
        Dictionary with keys like 'LEAR_1456', values are (prediction, metadata) tuples
    """
    total_start_time = time.time()
    total_windows = len(calibration_windows)
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"LEAR Forecast Batch Processing")
        print(f"Total calibration windows: {total_windows}")
        print(f"Windows: {calibration_windows}")
        print(f"{'#'*70}\n")
    
    results = {}
    for idx, cw in enumerate(calibration_windows, 1):
        key = f'LEAR_{cw}'
        if verbose:
            print(f"[{idx}/{total_windows}] Processing calibration window: {cw} days")
        
        prediction, metadata = run_lear_forecast(
            df_train=df_train,
            df_test=df_test,
            target_date=target_date,
            calibration_window=cw,
            alpha_fixed=alpha_fixed,
            reload_lear=(cw == calibration_windows[0]),  # Only reload for first
            verbose=verbose
        )
        results[key] = (prediction, metadata)
    
    total_elapsed = time.time() - total_start_time
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"LEAR Batch Processing Complete")
        print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        print(f"Average time per window: {total_elapsed/total_windows:.2f} seconds")
        print(f"{'#'*70}\n")
    
    return results


# ============================================================================
# DNN Model Functions
# ============================================================================

def run_dnn_hyperparameter_optimization(
    dataset: str = 'PJM',
    years_test: int = 2,
    calibration_window: int = 4,
    nlayers: int = 3,
    shuffle_train: bool = True,
    data_augmentation: bool = True,
    max_evals: int = 50,
    experiment_id: str = 'oneshot_dnn',
    path_datasets_folder: Union[str, Path] = 'datasets',
    path_hyperparameters_folder: Union[str, Path] = 'experimental_files',
    force_rerun: bool = False,
    early_stopping_patience: Optional[int] = None,
    batch_size: int = 5,
    min_improvement: float = 0.01
) -> Path:
    """
    Run hyperparameter optimization for DNN model with optional early stopping.
    
    Parameters:
    -----------
    dataset : str
        Dataset name (default: 'PJM')
    years_test : int
        Number of test years (default: 2)
    calibration_window : int
        Calibration window in years (default: 4)
    nlayers : int
        Number of layers (default: 3)
    shuffle_train : bool
        Whether to shuffle training data (default: True)
    data_augmentation : bool
        Whether to use data augmentation (default: True)
    max_evals : int
        Maximum hyperparameter search evaluations (default: 50)
    experiment_id : str
        Experiment identifier (default: 'oneshot_dnn')
    path_datasets_folder : str or Path
        Path to datasets folder
    path_hyperparameters_folder : str or Path
        Path to save hyperparameters
    force_rerun : bool
        Force rerun even if trials file exists (default: False)
    early_stopping_patience : int, optional
        Number of batches without improvement before stopping (default: None = no early stopping)
    batch_size : int
        Number of evaluations per batch when using early stopping (default: 5)
    min_improvement : float
        Minimum MAE improvement to count as improvement (default: 0.01)
    
    Returns:
    --------
    Path
        Path to the trials file
    """
    path_hyperparameters_folder = Path(path_hyperparameters_folder)
    path_hyperparameters_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate trials file name (must match epftoolbox naming convention exactly)
    # Note: epftoolbox does NOT include max_evals in the filename
    trials_file_name = (
        f'DNN_hyperparameters_nl{nlayers}_dat{dataset}_YT{years_test}'
        f"{'_SF' if shuffle_train else ''}"
        f"{'_DA' if data_augmentation else ''}"
        f'_CW{calibration_window}_{experiment_id}'
    )
    trials_path = path_hyperparameters_folder / trials_file_name
    
    if trials_path.exists() and not force_rerun:
        print(f'Trials file already exists, skipping hyperparameter search:')
        print(trials_path)
        return trials_path
    
    # Early stopping logic
    if early_stopping_patience is not None:
        print(f'Running hyperparameter search with early stopping...')
        print(f'  Batch size: {batch_size}')
        print(f'  Patience: {early_stopping_patience} batches without improvement')
        print(f'  Minimum improvement threshold: {min_improvement:.4f} MAE')
        
        best_test_mae = float('inf')
        no_improvement_count = 0
        current_evals = 0
        new_hyperopt = True
        
        while current_evals < max_evals:
            # Calculate how many evals to run in this batch
            remaining_evals = max_evals - current_evals
            batch_evals = min(batch_size, remaining_evals)
            
            print(f'\nRunning batch: {current_evals + 1}-{current_evals + batch_evals} of {max_evals}')
            
            # Run hyperparameter optimization for this batch
            hyperparameter_optimizer(
                path_datasets_folder=str(path_datasets_folder),
                path_hyperparameters_folder=str(path_hyperparameters_folder),
                new_hyperopt=new_hyperopt,
                max_evals=current_evals + batch_evals,
                nlayers=nlayers,
                dataset=dataset,
                years_test=years_test,
                calibration_window=calibration_window,
                shuffle_train=shuffle_train,
                data_augmentation=data_augmentation,
                experiment_id=experiment_id,
                begin_test_date=None,
                end_test_date=None,
            )
            
            # Wait a moment for file to be written
            time.sleep(0.5)
            
            # Load trials to check best test MAE
            try:
                if not trials_path.exists():
                    # Try to find the file - sometimes it takes a moment to appear
                    time.sleep(1)
                
                if trials_path.exists():
                    trials = pc.load(open(trials_path, "rb"))
                    # Find best test MAE from completed trials
                    completed_trials = [t for t in trials.trials if t.get('result') is not None]
                    if completed_trials:
                        current_best_mae = min(
                            t['result'].get('MAE Test', float('inf')) 
                            for t in completed_trials
                        )
                        
                        print(f'  Current best test MAE: {current_best_mae:.4f}')
                        
                        # Check if improvement is significant (above threshold)
                        if best_test_mae == float('inf'):
                            # First batch - just record the value
                            best_test_mae = current_best_mae
                            print(f'  Initial best test MAE: {best_test_mae:.4f}')
                            no_improvement_count = 0
                        elif current_best_mae < (best_test_mae - min_improvement):
                            improvement = best_test_mae - current_best_mae
                            print(f'  ✓ Improved by {improvement:.4f} (above threshold)')
                            best_test_mae = current_best_mae
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            if current_best_mae < best_test_mae:
                                improvement = best_test_mae - current_best_mae
                                print(f'  ⚠ Marginal improvement {improvement:.4f} (below threshold {min_improvement:.4f})')
                            else:
                                print(f'  ⚠ No improvement ({no_improvement_count}/{early_stopping_patience})')
                            
                            if no_improvement_count >= early_stopping_patience:
                                print(f'\nEarly stopping triggered: No significant improvement for {early_stopping_patience} batches')
                                print(f'Best test MAE achieved: {best_test_mae:.4f}')
                                break
                    else:
                        print(f'  Warning: No completed trials found')
                else:
                    print(f'  Warning: Trials file not found at {trials_path}')
            except Exception as e:
                print(f'  Warning: Could not read trials file: {e}')
                import traceback
                traceback.print_exc()
            
            current_evals += batch_evals
            new_hyperopt = False  # Continue from existing trials
        
        print(f'\nHyperparameter search completed: {current_evals}/{max_evals} evaluations')
    else:
        print('Running hyperparameter search...')
        hyperparameter_optimizer(
            path_datasets_folder=str(path_datasets_folder),
            path_hyperparameters_folder=str(path_hyperparameters_folder),
            new_hyperopt=True,
            max_evals=max_evals,
            nlayers=nlayers,
            dataset=dataset,
            years_test=years_test,
            calibration_window=calibration_window,
            shuffle_train=shuffle_train,
            data_augmentation=data_augmentation,
            experiment_id=experiment_id,
            begin_test_date=None,
            end_test_date=None,
        )
    
    print(f'Saved trials to: {trials_path}')
    return trials_path


def run_dnn_forecast(
    experiment_id: str,
    target_date: Union[str, pd.Timestamp],
    path_datasets_folder: Union[str, Path] = 'datasets',
    path_hyperparameter_folder: Union[str, Path] = 'experimental_files',
    path_recalibration_folder: Union[str, Path] = 'forecasts_local',
    dataset: str = 'PJM',
    years_test: int = 2,
    new_recalibration: bool = True,
    nlayers: int = 2,
    shuffle_train: bool = True,
    data_augmentation: int = 0,
    calibration_window: int = 4,
) -> pd.DataFrame:
    """
    Run DNN forecast for a single day.
    
    Parameters:
    -----------
    experiment_id : str
        Experiment identifier (must match hyperparameter optimization)
    target_date : str or pd.Timestamp
        Target date for forecast
    path_datasets_folder : str or Path
        Path to datasets folder
    path_hyperparameter_folder : str or Path
        Path to hyperparameters folder
    path_recalibration_folder : str or Path
        Path to save forecasts
    dataset : str
        Dataset name (default: 'PJM')
    years_test : int
        Number of test years (default: 2)
    new_recalibration : bool
        Force fresh recalibration (default: True)
    nlayers : int
        Number of hidden layers; must match the hyperparameter search run
    shuffle_train : bool
        Whether shuffling was used during hyperparameter search
    data_augmentation : int
        Whether data augmentation was used (0/1) in hyperparameter search
    calibration_window : int
        Calibration window in years; must match the hyperparameter search
    
    Returns:
    --------
    pd.DataFrame
        Forecast dataframe with 24-hour predictions
    """
    target_date = pd.to_datetime(target_date)
    path_recalibration_folder = Path(path_recalibration_folder)
    path_recalibration_folder.mkdir(parents=True, exist_ok=True)
    
    # Format dates for epftoolbox (DD/MM/YYYY HH:MM)
    begin_test_date = target_date.strftime('%d/%m/%Y %H:%M')
    end_test_date = begin_test_date  # Single day trick
    
    forecast_dnn = evaluate_dnn_in_test_dataset(
        experiment_id=experiment_id,
        path_datasets_folder=str(path_datasets_folder),
        path_hyperparameter_folder=str(path_hyperparameter_folder),
        path_recalibration_folder=str(path_recalibration_folder),
        dataset=dataset,
        years_test=years_test,
        nlayers=nlayers,
        shuffle_train=shuffle_train,
        data_augmentation=data_augmentation,
        calibration_window=calibration_window,
        begin_test_date=begin_test_date,
        end_test_date=end_test_date,
        new_recalibration=new_recalibration,
    )
    
    return forecast_dnn


def run_dnn_forecast_multiday(
    experiment_id: str,
    target_start_date: Union[str, pd.Timestamp],
    n_days: int,
    path_datasets_folder: Union[str, Path] = 'datasets',
    path_hyperparameter_folder: Union[str, Path] = 'experimental_files',
    path_recalibration_folder: Union[str, Path] = 'forecasts_local',
    dataset: str = 'PJM',
    years_test: int = 2,
    new_recalibration: bool = True,
    verbose: bool = True,
    nlayers: int = 2,
    shuffle_train: bool = True,
    data_augmentation: int = 0,
    calibration_window: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Run DNN forecasts for multiple consecutive days with daily recalibration.
    
    Parameters:
    -----------
    experiment_id : str
        Experiment identifier (must match hyperparameter optimization)
    target_start_date : str or pd.Timestamp
        Start date for forecasts
    n_days : int
        Number of consecutive days to forecast
    path_datasets_folder : str or Path
        Path to datasets folder
    path_hyperparameter_folder : str or Path
        Path to hyperparameters folder
    path_recalibration_folder : str or Path
        Path to save forecasts
    dataset : str
        Dataset name (default: 'PJM')
    years_test : int
        Number of test years (default: 2)
    new_recalibration : bool
        Force fresh recalibration for each day (default: True)
    verbose : bool
        Whether to print progress information (default: True)
    nlayers : int
        Number of hidden layers; must match the hyperparameter search run
    shuffle_train : bool
        Whether shuffling was used during hyperparameter search
    data_augmentation : int
        Whether data augmentation was used (0/1) in hyperparameter search
    calibration_window : int
        Calibration window in years; must match the hyperparameter search
    
    Returns:
    --------
    dict
        Dictionary mapping date strings to forecast DataFrames
    """
    target_start_date = pd.to_datetime(target_start_date)
    all_forecasts = {}
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"DNN Multi-Day Forecast Processing")
        print(f"Start Date: {target_start_date.date()}")
        print(f"Number of Days: {n_days}")
        print(f"{'#'*70}\n")
    
    for day_idx in range(n_days):
        current_date = target_start_date + pd.Timedelta(days=day_idx)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Day {day_idx + 1}/{n_days}: {current_date.date()}")
            print(f"{'='*70}")
        
        forecast_dnn = run_dnn_forecast(
            experiment_id=experiment_id,
            target_date=current_date,
            path_datasets_folder=path_datasets_folder,
            path_hyperparameter_folder=path_hyperparameter_folder,
            path_recalibration_folder=path_recalibration_folder,
            dataset=dataset,
            years_test=years_test,
            new_recalibration=new_recalibration,
            nlayers=nlayers,
            shuffle_train=shuffle_train,
            data_augmentation=data_augmentation,
            calibration_window=calibration_window,
        )
        
        all_forecasts[current_date.strftime('%Y-%m-%d')] = forecast_dnn
        
        if verbose:
            print(f"✓ Completed forecast for {current_date.date()}")
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"Multi-Day DNN Forecast Complete")
        print(f"Total days processed: {n_days}")
        print(f"{'#'*70}\n")
    
    return all_forecasts


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
    df_train: pd.DataFrame,
    target_date: Union[str, pd.Timestamp]
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual price values (24 hours)
    predictions : np.ndarray
        Predicted price values (24 hours)
    df_train : pd.DataFrame
        Training dataframe (for MASE calculation)
    target_date : str or pd.Timestamp
        Target date (for datetime index)
    
    Returns:
    --------
    dict
        Dictionary of metric names and values
    """
    target_date = pd.to_datetime(target_date)
    datetime_index = pd.date_range(
        start=target_date, periods=24, freq='1h'
    )
    
    # Convert to DataFrames for MASE
    actual_df = pd.DataFrame(actual, index=datetime_index, columns=['Price'])
    predictions_df = pd.DataFrame(
        predictions, index=datetime_index, columns=['Price']
    )
    p_real_in = df_train[['Price']]
    
    # Calculate metrics
    mae_val = MAE(p_real=actual, p_pred=predictions)
    rmse_val = RMSE(p_real=actual, p_pred=predictions)
    mape_val = MAPE(p_real=actual, p_pred=predictions) * 100
    smape_val = sMAPE(p_real=actual, p_pred=predictions) * 100
    mase_val = MASE(
        p_real=actual_df,
        p_pred=predictions_df,
        p_real_in=p_real_in,
        m=None
    )
    
    # Calculate rMAE manually (one-step ahead naive forecast)
    naive_mae = np.mean(np.abs(actual[1:] - actual[:-1]))
    if naive_mae > 0:
        rmae_val = mae_val / naive_mae
    else:
        rmae_val = np.nan
    
    return {
        'MAE': mae_val,
        'RMSE': rmse_val,
        'MAPE (%)': mape_val,
        'sMAPE (%)': smape_val,
        'MASE': mase_val,
        'rMAE': rmae_val
    }


def evaluate_all_models(
    actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    df_train: pd.DataFrame,
    target_date: Union[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Evaluate multiple models and return metrics dataframe.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual price values
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays
    df_train : pd.DataFrame
        Training dataframe
    target_date : str or pd.Timestamp
        Target date
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with metrics for each model
    """
    results = {}
    for model_name, predictions in predictions_dict.items():
        metrics = calculate_metrics(
            actual=actual,
            predictions=predictions,
            df_train=df_train,
            target_date=target_date
        )
        results[model_name] = metrics
    
    metrics_df = pd.DataFrame(results).T
    return metrics_df


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_predictions(
    predictions_dict: Dict[str, np.ndarray],
    target_date: Union[str, pd.Timestamp],
    output_path: Union[str, Path] = 'forecasts_local',
    filename: Optional[str] = None
) -> Path:
    """
    Save predictions to CSV file.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary mapping model names to prediction arrays
    target_date : str or pd.Timestamp
        Target date
    output_path : str or Path
        Output directory
    filename : str, optional
        Custom filename (if None, auto-generates)
    
    Returns:
    --------
    Path
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_date = pd.to_datetime(target_date)
    if filename is None:
        filename = f'predictions_{target_date.strftime("%Y%m%d")}.csv'
    
    filepath = output_path / filename
    
    # Create dataframe with hours as columns
    data = {}
    for model_name, predictions in predictions_dict.items():
        data[model_name] = predictions
    
    df = pd.DataFrame(data)
    df.index.name = 'Hour'
    df.to_csv(filepath)
    
    print(f"Saved predictions to: {filepath}")
    return filepath


def load_published_benchmarks(
    csv_path: Union[str, Path] = 'forecasts/Forecasts_PJM_DNN_LEAR_ensembles.csv',
    target_date: Union[str, pd.Timestamp] = '2016-12-27',
    model_columns: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Load published benchmark forecasts from CSV.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to benchmark CSV file
    target_date : str or pd.Timestamp
        Target date to extract
    model_columns : list of str, optional
        List of model column names to extract (if None, extracts common ones)
    
    Returns:
    --------
    dict
        Dictionary mapping model names to prediction lists
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    target_date = pd.to_datetime(target_date)
    
    # Find rows matching target date
    date_col = df.columns[0]  # Usually first column
    matching_rows = df[df[date_col].str.startswith(str(target_date.date()))]
    
    if model_columns is None:
        # Default model columns
        model_columns = ['LEAR 56', 'LEAR 84', 'LEAR 1092', 'LEAR 1456',
                        'DNN 1', 'DNN 2', 'DNN 3', 'DNN 4']
    
    results = {}
    for col in model_columns:
        if col in df.columns:
            # Extract first 24 values (or all if less)
            values = matching_rows[col].iloc[:24].tolist()
            results[col] = values
    
    return results

