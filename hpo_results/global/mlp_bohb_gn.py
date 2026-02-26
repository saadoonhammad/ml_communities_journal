# ============================================================================
# MLP AUTOENCODER - BAYESIAN HYPERBAND OPTIMIZATION (BOHB) - FUSED MODEL
# ============================================================================
"""
Bayesian Hyperband Optimization for MLP Autoencoder using Optuna.
Combines Tree-structured Parzen Estimator (TPE) with Hyperband pruning
for efficient hyperparameter search in multi-sensor anomaly detection.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import json
import joblib
import warnings
import os
import random
import matplotlib.pyplot as plt
import logging
import gc
import sys
from typing import Dict, List, Tuple, Any

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
RANDOM_SEED = 42

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'../../Logs/global_mlp_bohb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================
CLUSTER = 'all'

# Sensor columns (same as in your fused model)
SENSOR_COLS = ['c01m061e01', 'c01m127e01', 'c01m141e01', 'c02m014e04','c02m042e02', 'c02m103e02', 
               'c02m119e02', 'c04m072e01', 'c05m105e08','c07m114e01', 'c08m131e01', 'c03m102e01',
               'c05m031e06', 'c05m040e13', 'c05m050e01', 'c05m117e01', 'c05m902e01', 'c06m011e04', 
               'c06m032e07', 'c06m084e03', 'c06m136e01', 'c06m136e02', 'c03m070e01', 'c03m096e01',
               'c03m096e03', 'c04m122e01', 'c05m105e09', 'c05m128e02', 'c06m002e01', 'c06m057e02', 
               'c07m065e01', 'c08m017e01', 'c08m046e01', 'c01m080e01','c01m129e02', 'c02m051e03', 
               'c02m139e04', 'c03m093e04']

# Data path
TRAIN_CSV_PATH = f'../../Global_Network/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_PREFIX = f'{CLUSTER}_global_mlp_bohb_results'

logger.info(f"="*80)
logger.info(f"Configured for CLUSTER: {CLUSTER}")
logger.info(f"Data path: {TRAIN_CSV_PATH}")
logger.info(f"Sensors: {SENSOR_COLS}")
logger.info(f"Output prefix: {OUTPUT_PREFIX}")
logger.info(f"="*80)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
TARGET_GPU = 0

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[TARGET_GPU], 'GPU')
        logger.info(f"✓ Using GPU {TARGET_GPU}: {gpus[TARGET_GPU].name}")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")

tf.config.experimental.enable_op_determinism()
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger.info("="*80)
logger.info("MLP AUTOENCODER - BAYESIAN HYPERBAND OPTIMIZATION (BOHB)")
logger.info("="*80)
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Optuna version: {optuna.__version__}")
logger.info(f"Random seed: {RANDOM_SEED}")
logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """Create sliding window sequences preserving chronological order."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def prepare_multisensor_data(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, List[str]]:
    """Prepare multi-sensor data for MLP autoencoder."""
    logger.info(f"Preparing multi-sensor data")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Extract sensor data
    sensor_data = df[SENSOR_COLS].values
    
    # Handle missing values
    if np.isnan(sensor_data).any():
        logger.warning("Missing values detected - filling with forward fill")
        df[SENSOR_COLS] = df[SENSOR_COLS].fillna(method='ffill')
        sensor_data = df[SENSOR_COLS].values
    
    # Create sequences
    sequences = create_sequences(sensor_data, sequence_length)
    
    logger.info(f"Sequences shape: {sequences.shape}")
    logger.info(f"Sensors ({sequences.shape[2]}): {SENSOR_COLS}")
    
    return sequences, SENSOR_COLS

def create_expanding_window_splits(
    data: pd.DataFrame, 
    sequence_length: int, 
    n_folds: int = 5, 
    val_ratio: float = 0.20, 
    use_purge: bool = True, 
    purge_mult: int = 2
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding window splits for multi-sensor data."""
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    N = len(data_sorted)
    
    if N <= sequence_length:
        raise ValueError(f"Data too small. Need at least {sequence_length + 1} points.")
    
    purge = (purge_mult * sequence_length) if use_purge else 0
    total_span = N - sequence_length
    init_val = min(int(total_span * val_ratio), total_span - n_folds)
    max_val_by_formula = total_span // (n_folds + 1)
    val_size = min(init_val, max_val_by_formula)
    
    step = (total_span - val_size) // n_folds
    required_step = max(1, val_size + purge - sequence_length)
    
    while step < required_step and val_size > 10:
        val_size -= 1
        step = (total_span - val_size) // n_folds
        required_step = max(1, val_size + purge - sequence_length)
    
    if step < 1:
        raise ValueError("Cannot create folds with step>=1. Try reducing n_folds or val_ratio.")
    
    logger.info(f"Creating {n_folds} expanding window folds...")
    logger.info(f"Total usable samples: {total_span}")
    logger.info(f"Validation size per fold: {val_size}")
    logger.info(f"Purge gap per fold: {purge} (purge_mult={purge_mult})")
    logger.info(f"Training step per fold: {step}")
    
    splits = []
    prev_train_end = -1
    
    for fold in range(1, n_folds + 1):
        val_start_raw = sequence_length + fold * step
        train_end_idx = max(val_start_raw - purge, sequence_length)
        val_start_idx = val_start_raw
        val_end_idx = val_start_idx + val_size
        
        if val_end_idx > N:
            raise ValueError(f"Fold {fold} exceeds data bounds.")
        
        if fold > 1 and train_end_idx <= prev_train_end:
            raise ValueError(f"Fold {fold} violates expanding window.")
        prev_train_end = train_end_idx
        
        train_chunk = data_sorted.iloc[:train_end_idx].copy()
        val_chunk = data_sorted.iloc[val_start_idx:val_end_idx].copy()
        
        if len(train_chunk) < len(val_chunk):
            raise ValueError(f"Fold {fold}: train size < val size.")
        
        logger.info(f"Fold {fold}:")
        logger.info(f"  - Train: indices 0 to {train_end_idx-1} ({len(train_chunk)} pts)")
        if purge > 0:
            gap_size = val_start_idx - train_end_idx
            logger.info(f"  - Gap  : indices {train_end_idx} to {val_start_idx-1} ({gap_size} pts)")
        logger.info(f"  - Val  : indices {val_start_idx} to {val_end_idx-1} ({len(val_chunk)} pts)")
        
        splits.append((train_chunk, val_chunk))
    
    return splits

# ============================================================================
# MLP AUTOENCODER MODEL CLASS
# ============================================================================

class FusedMLPAutoencoder:
    """Multi-Layer Perceptron Autoencoder for multi-sensor time series."""
    
    def __init__(self, sequence_length: int, n_sensors: int,
                 hidden_units: int = 64, encoding_dim: int = 32, 
                 learning_rate: float = 0.001, dropout_rate: float = 0.2):
        self.sequence_length = sequence_length
        self.n_sensors = n_sensors
        self.hidden_units = hidden_units
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the MLP Autoencoder architecture."""
        inp = Input(shape=(self.sequence_length, self.n_sensors))
        
        # Flatten input
        x = Flatten()(inp)
        
        # Encoder
        x = Dense(self.hidden_units * 2, activation='tanh')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.hidden_units, activation='tanh')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Bottleneck
        z = Dense(self.encoding_dim, activation='tanh')(x)
        
        # Decoder
        x = Dense(self.hidden_units, activation='tanh')(z)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.hidden_units * 2, activation='tanh')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output
        x = Dense(self.sequence_length * self.n_sensors)(x)
        out = Reshape((self.sequence_length, self.n_sensors))(x)
        
        self.model = Model(inp, out)
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray, 
              epochs: int = 100, batch_size: int = 32, 
              patience: int = 15, callbacks: list = None) -> Any:
        """Train the autoencoder on multi-sensor data."""
        if self.model is None:
            self.build_model()
        
        default_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=max(5, patience//3), 
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        self.history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            verbose=0,
            shuffle=False
        )
        
        return self.history
    
    def cleanup(self):
        """Free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        tf.keras.backend.clear_session()
        gc.collect()

# ============================================================================
# OPTUNA CUSTOM CALLBACK FOR PRUNING
# ============================================================================

class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Callback to prune unpromising trials in Optuna."""
    
    def __init__(self, trial: optuna.Trial, monitor: str = 'val_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Report intermediate value to Optuna
        self.trial.report(current_value, epoch)
        
        # Prune if needed
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned()

# ============================================================================
# BAYESIAN HYPERBAND OPTIMIZER CLASS
# ============================================================================

class BayesianHyperbandOptimizer:
    """
    Bayesian Hyperband Optimization (BOHB) for MLP Autoencoder.
    Combines TPE sampling with Hyperband pruning for efficient HPO.
    """
    
    def __init__(self, data_splits: List[Tuple[pd.DataFrame, pd.DataFrame]], 
                 sequence_length: int, n_trials: int = 50, 
                 min_epochs: int = 20, max_epochs: int = 100, 
                 eta: int = 3, random_state: int = 42):
        """
        Initialize BOHB optimizer.
        
        Args:
            data_splits: CV splits as (train_df, val_df) tuples
            sequence_length: Input sequence length
            n_trials: Total number of hyperparameter trials
            min_epochs: Minimum training epochs (resource level)
            max_epochs: Maximum training epochs (resource level)
            eta: Downsampling factor for Hyperband (typical: 2 or 3)
            random_state: Random seed for reproducibility
        """
        self.data_splits = data_splits
        self.sequence_length = sequence_length
        self.n_trials = n_trials
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.eta = eta
        self.random_state = random_state
        
        self.best_loss = float('inf')
        self.best_params = None
        self.all_results = []
        self.study = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Returns the average validation loss across CV folds.
        """
        # Define hyperparameter search space
        params = {
            'hidden_units': trial.suggest_int('hidden_units', 64, 256, log=True),
            'encoding_dim': trial.suggest_int('encoding_dim', 16, 96, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
            'epochs': trial.suggest_int('epochs', self.min_epochs, self.max_epochs),
            'patience_factor': trial.suggest_float('patience_factor', 0.1, 0.3)
        }
        
        logger.info(f"\nTrial {trial.number + 1}/{self.n_trials}")
        logger.info(f"Testing params: {params}")
        
        fold_val_losses = []
        fold_train_times = []
        
        try:
            for fold_idx, (df_train_fold, df_val_fold) in enumerate(self.data_splits):
                logger.info(f"  Fold {fold_idx + 1}/{len(self.data_splits)}")
                fold_start = datetime.now()
                
                # Skip if insufficient data
                if len(df_train_fold) <= self.sequence_length or len(df_val_fold) <= self.sequence_length:
                    logger.warning(f"    Insufficient data - skipping")
                    continue
                
                # Prepare data
                X_train, _ = prepare_multisensor_data(df_train_fold, self.sequence_length)
                X_val, _ = prepare_multisensor_data(df_val_fold, self.sequence_length)
                
                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"    No sequences created - skipping")
                    continue
                
                n_sensors = X_train.shape[2]
                
                # Scale each sensor independently
                scalers = []
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                
                for i in range(n_sensors):
                    scaler = StandardScaler()
                    X_train_scaled[:, :, i] = scaler.fit_transform(
                        X_train[:, :, i].reshape(-1, 1)
                    ).reshape(X_train.shape[0], self.sequence_length)
                    X_val_scaled[:, :, i] = scaler.transform(
                        X_val[:, :, i].reshape(-1, 1)
                    ).reshape(X_val.shape[0], self.sequence_length)
                    scalers.append(scaler)
                
                # Build model
                autoencoder = FusedMLPAutoencoder(
                    sequence_length=self.sequence_length,
                    n_sensors=n_sensors,
                    hidden_units=params['hidden_units'],
                    encoding_dim=params['encoding_dim'],
                    learning_rate=params['learning_rate'],
                    dropout_rate=params['dropout_rate']
                )
                
                # Calculate patience for early stopping
                patience = max(5, int(params['patience_factor'] * params['epochs']))
                
                # Train with pruning callback
                callbacks = [OptunaPruningCallback(trial, monitor='val_loss')]
                
                history = autoencoder.train(
                    X_train_scaled, X_val_scaled,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    patience=patience,
                    callbacks=callbacks
                )
                
                # Get best validation loss
                val_losses = history.history['val_loss']
                valid_losses = [v for v in val_losses if not (np.isnan(v) or np.isinf(v))]
                
                if valid_losses:
                    best_val_loss = min(valid_losses)
                    fold_val_losses.append(best_val_loss)
                    
                    # Report intermediate value for pruning
                    trial.report(best_val_loss, step=fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                fold_time = (datetime.now() - fold_start).total_seconds()
                fold_train_times.append(fold_time)
                
                logger.info(f"    Val Loss: {best_val_loss:.6f}, Time: {fold_time:.1f}s")
                
                # Cleanup
                autoencoder.cleanup()
                del X_train, X_val, X_train_scaled, X_val_scaled, scalers
                gc.collect()
            
            if not fold_val_losses:
                logger.warning("  No valid folds completed - returning penalty")
                return float('inf')
            
            avg_val_loss = np.mean(fold_val_losses)
            avg_train_time = np.mean(fold_train_times)
            
            logger.info(f"  Average Val Loss: {avg_val_loss:.6f}")
            logger.info(f"  Average Train Time: {avg_train_time:.1f}s")
            
            # Store results
            self.all_results.append({
                'trial_number': trial.number,
                'params': params,
                'avg_val_loss': avg_val_loss,
                'fold_losses': fold_val_losses,
                'avg_train_time': avg_train_time,
                'n_folds_completed': len(fold_val_losses),
                'state': 'COMPLETE'
            })
            
            # Update best
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_params = params.copy()
                logger.info(f"  ✓ NEW BEST! Loss: {avg_val_loss:.6f}")
            
            return avg_val_loss
            
        except optuna.TrialPruned:
            logger.info(f"  Trial pruned early")
            self.all_results.append({
                'trial_number': trial.number,
                'params': params,
                'avg_val_loss': None,
                'state': 'PRUNED'
            })
            raise
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"  GPU OOM Error: {str(e)}")
            return float('inf')
            
        except Exception as e:
            logger.error(f"  Unexpected error: {str(e)}")
            return float('inf')
            
        finally:
            tf.keras.backend.clear_session()
            gc.collect()
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run Bayesian Hyperband optimization.
        """
        logger.info(f"\n{'='*80}")
        logger.info("STARTING BAYESIAN HYPERBAND OPTIMIZATION (BOHB)")
        logger.info(f"{'='*80}")
        logger.info(f"Total trials: {self.n_trials}")
        logger.info(f"Resource range: {self.min_epochs}-{self.max_epochs} epochs")
        logger.info(f"Downsampling rate (eta): {self.eta}")
        logger.info(f"CV folds: {len(self.data_splits)}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Random seed: {self.random_state}")
        logger.info(f"{'='*80}\n")
        
        # Create sampler and pruner for BOHB
        sampler = TPESampler(
            n_startup_trials=10,  # Number of random trials before Bayesian
            n_ei_candidates=24,
            seed=self.random_state
        )
        
        pruner = HyperbandPruner(
            min_resource=self.min_epochs,
            max_resource=self.max_epochs,
            reduction_factor=self.eta,
            bootstrap_count=5
        )
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"mlp_bohb_{CLUSTER}",
            storage=None,  # In-memory storage
            load_if_exists=False
        )
        
        start_time = datetime.now()
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=None,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        duration = datetime.now() - start_time
        
        # Get best results
        self.best_loss = self.study.best_value
        self.best_params = self.study.best_params
        
        # Count trials
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {duration}")
        logger.info(f"Best validation loss: {self.best_loss:.6f}")
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Trials: {completed_trials} completed, {pruned_trials} pruned")
        logger.info(f"{'='*80}\n")
        
        return {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'study': self.study,
            'all_results': self.all_results,
            'duration': duration.total_seconds(),
            'trial_stats': {
                'total': len(self.study.trials),
                'completed': completed_trials,
                'pruned': pruned_trials
            }
        }

# ============================================================================
# VISUALIZATION AND SAVING FUNCTIONS
# ============================================================================

def save_bohb_results(results: Dict[str, Any], output_dir: str = '../../HPO Results/Network_Global/global_mlp_bohb_results'):
    """Save BOHB optimization results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{OUTPUT_PREFIX}_{timestamp}"
    
    # Save study object
    study_file = os.path.join(output_dir, f"{base_filename}_study.pkl")
    joblib.dump(results['study'], study_file)
    logger.info(f"✓ Study saved to {study_file}")
    
    # Save results as JSON
    results_file = os.path.join(output_dir, f"{base_filename}_results.json")
    
    # Prepare JSON-serializable results
    json_results = {
        'optimization_method': 'Bayesian_Hyperband_BOHB',
        'cluster': CLUSTER,
        'best_params': results['best_params'],
        'best_loss': float(results['best_loss']),
        'duration_seconds': results['duration'],
        'hyperband_config': {
            'min_epochs': 20,
            'max_epochs': 100,
            'eta': 3,
            'n_trials': 50
        },
        'timestamp': datetime.now().isoformat(),
        'sensors': SENSOR_COLS,
        'trial_statistics': results['trial_stats'],
        'environment': {
            'tensorflow_version': tf.__version__,
            'optuna_version': optuna.__version__,
            'numpy_version': np.__version__,
            'python_version': sys.version,
            'random_seed': RANDOM_SEED
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"✓ Results saved to {results_file}")
    
    # Generate convergence plots
    plot_file = os.path.join(output_dir, f"{base_filename}_convergence.png")
    plot_bohb_convergence(results, plot_file)
    
    # Generate parameter importance plot
    importance_file = os.path.join(output_dir, f"{base_filename}_importance.png")
    plot_parameter_importance(results['study'], importance_file)
    
    return {
        'study_file': study_file,
        'results_file': results_file,
        'plot_file': plot_file,
        'importance_file': importance_file
    }

def plot_bohb_convergence(results: Dict[str, Any], filename: str):
    """Plot optimization convergence for BOHB."""
    study = results['study']
    
    # Get valid trials (not infinite loss)
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value < 1e10]
    
    if not valid_trials:
        logger.warning("No valid trials to plot")
        return
    
    # Extract values
    values = [t.value for t in valid_trials]
    trial_numbers = [t.number for t in valid_trials]
    
    # Sort by trial number
    sorted_indices = np.argsort(trial_numbers)
    values_sorted = np.array(values)[sorted_indices]
    trial_numbers_sorted = np.array(trial_numbers)[sorted_indices]
    
    # Calculate cumulative minimum
    cumulative_min = np.minimum.accumulate(values_sorted)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence
    ax1.scatter(trial_numbers_sorted, values_sorted, alpha=0.6, s=20, label='Trial Loss')
    ax1.plot(trial_numbers_sorted, cumulative_min, 'r-', linewidth=2, label='Best Loss')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax1.set_title('BOHB Optimization Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss distribution
    ax2.hist(values_sorted, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(results['best_loss'], color='red', linestyle='--', 
               linewidth=2, label=f'Best: {results["best_loss"]:.6f}')
    ax2.set_xlabel('Validation Loss (MSE)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Convergence plot saved to {filename}")

def plot_parameter_importance(study: optuna.Study, filename: str):
    """Plot hyperparameter importance from Optuna study."""
    try:
        importances = optuna.importance.get_param_importances(study)
        
        if not importances:
            logger.warning("Could not compute parameter importances")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(importances.keys())
        importance_vals = list(importances.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance_vals)
        params = [params[i] for i in sorted_idx]
        importance_vals = [importance_vals[i] for i in sorted_idx]
        
        # Create horizontal bar plot
        bars = ax.barh(params, importance_vals, color='steelblue', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, importance_vals):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Hyperparameter Importance (BOHB)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Parameter importance plot saved to {filename}")
        
    except Exception as e:
        logger.warning(f"Could not create importance plot: {str(e)}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_bayesian_hyperband_optimization(
    csv_path: str = None,
    sequence_length: int = 24,
    n_trials: int = 50,
    n_folds: int = 5,
    min_epochs: int = 20,
    max_epochs: int = 100,
    eta: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run complete Bayesian Hyperband optimization pipeline.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("="*80)
    
    if csv_path is None:
        csv_path = TRAIN_CSV_PATH
    
    if not os.path.exists(csv_path):
        logger.error(f"ERROR: '{csv_path}' not found.")
        return None
    
    # Load data
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for missing values
    for sensor in SENSOR_COLS:
        if sensor in df.columns and df[sensor].isna().any():
            logger.warning(f"Missing values in {sensor} - filling with forward fill")
            df[sensor].fillna(method='ffill', inplace=True)
    
    logger.info(f"✓ Loaded data: {len(df)} samples")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CREATE EXPANDING WINDOW CV SPLITS")
    logger.info("="*80)
    
    try:
        data_splits = create_expanding_window_splits(
            df, 
            sequence_length=sequence_length, 
            n_folds=n_folds
        )
        
        if not data_splits:
            logger.error("ERROR: No valid CV splits created")
            return None
            
        logger.info(f"✓ Created {len(data_splits)} CV splits")
        
    except Exception as e:
        logger.error(f"ERROR creating splits: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: RUN BAYESIAN HYPERBAND OPTIMIZATION")
    logger.info("="*80)
    
    optimizer = BayesianHyperbandOptimizer(
        data_splits=data_splits,
        sequence_length=sequence_length,
        n_trials=n_trials,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        eta=eta,
        random_state=random_state
    )
    
    start_time = datetime.now()
    results = optimizer.run_optimization()
    duration = datetime.now() - start_time
    
    logger.info(f"Total optimization time: {duration}")
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SAVE RESULTS")
    logger.info("="*80)
    
    saved_files = save_bohb_results(results)
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Cluster: {CLUSTER}")
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Number of sensors: {len(SENSOR_COLS)}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Best validation loss: {results['best_loss']:.6f}")
    logger.info(f"Best hyperparameters:")
    for k, v in results['best_params'].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Results file: {saved_files['results_file']}")
    logger.info("="*80)
    
    return {
        'results': results,
        'saved_files': saved_files,
        'config': {
            'sequence_length': sequence_length,
            'n_trials': n_trials,
            'n_folds': n_folds,
            'min_epochs': min_epochs,
            'max_epochs': max_epochs,
            'eta': eta
        }
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("MLP AUTOENCODER - BAYESIAN HYPERBAND OPTIMIZATION (BOHB)")
    logger.info("="*80)
    
    try:
        # Configuration (modify these as needed)
        SEQUENCE_LENGTH = 24
        N_TRIALS = 60
        N_FOLDS = 5
        MIN_EPOCHS = 30
        MAX_EPOCHS = 100
        ETA = 2
        RANDOM_STATE = 42
        
        if not os.path.exists(TRAIN_CSV_PATH):
            logger.error(f"ERROR: '{TRAIN_CSV_PATH}' not found.")
            logger.info("Please ensure the training data file exists at the specified path.")
            logger.info("Current working directory:", os.getcwd())
        else:
            # ================================================================
            logger.info("\n" + "="*80)
            logger.info("STARTING BAYESIAN HYPERBAND OPTIMIZATION")
            logger.info("="*80)
            logger.info(f"Configuration:")
            logger.info(f"  Sequence length: {SEQUENCE_LENGTH}")
            logger.info(f"  Number of trials: {N_TRIALS}")
            logger.info(f"  Epoch range: {MIN_EPOCHS}-{MAX_EPOCHS}")
            logger.info(f"  Eta (downsampling rate): {ETA}")
            logger.info(f"  CV folds: {N_FOLDS}")
            logger.info("="*80)
            
            results_dict = run_bayesian_hyperband_optimization(
                csv_path=TRAIN_CSV_PATH,
                sequence_length=SEQUENCE_LENGTH,
                n_trials=N_TRIALS,
                n_folds=N_FOLDS,
                min_epochs=MIN_EPOCHS,
                max_epochs=MAX_EPOCHS,
                eta=ETA,
                random_state=RANDOM_STATE
            )
            
            if results_dict is not None:
                logger.info("\n" + "="*80)
                logger.info("BAYESIAN HYPERBAND OPTIMIZATION COMPLETED SUCCESSFULLY!")
                logger.info("="*80)
                logger.info(f"Best validation loss: {results_dict['results']['best_loss']:.6f}")
                logger.info(f"Best parameters:")
                for k, v in results_dict['results']['best_params'].items():
                    logger.info(f"  {k}: {v}")
                
                # Show efficiency metrics
                total_trials = results_dict['results']['trial_stats']['total']
                pruned_trials = results_dict['results']['trial_stats']['pruned']
                efficiency = (pruned_trials / total_trials * 100) if total_trials > 0 else 0
                
                logger.info(f"\nEfficiency metrics:")
                logger.info(f"  Total trials attempted: {total_trials}")
                logger.info(f"  Trials pruned early: {pruned_trials}")
                logger.info(f"  Pruning efficiency: {efficiency:.1f}%")
                
            else:
                logger.error("Bayesian Hyperband optimization failed. Check logs for details.")
    
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user.")
        logger.info("Partial results have been logged.")
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("SCRIPT COMPLETED")
    logger.info("="*80)