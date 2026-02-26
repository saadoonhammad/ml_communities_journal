# ============================================================================
# MLP AUTOENCODER - BAYESIAN HYPERBAND OPTIMIZATION (BOHB)
# ============================================================================
"""
Bayesian Hyperband Optimization (BOHB) for MLP Autoencoder using Optuna.
Combines Tree-structured Parzen Estimator (TPE) with Hyperband pruning
for efficient hyperparameter search in time series anomaly detection.

Optimizes: hidden_units, encoding_dim, learning_rate, dropout_rate, batch_size
Objective: Minimize validation reconstruction error (MSE)
Validation: 5-fold expanding window cross-validation (chronological order preserved)
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
        logging.FileHandler(f'../Logs/mlp_bohb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================

CLUSTER = 'c3'  # Options: 'c1', 'c2', 'c3', 'c4'


CSV_PATH = f'../1. Datasets/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_PREFIX = f'{CLUSTER}_mlp_bohb_results'

logger.info(f"="*80)
logger.info(f"Configured for CLUSTER: {CLUSTER}")
logger.info(f"Data path: {CSV_PATH}")
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
# FEATURE ENGINEERING FUNCTIONS (KEEPING ORIGINAL)
# ============================================================================

def create_sequences(data, sequence_length):
    """Create sliding window sequences preserving chronological order."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def add_temporal_features(df):
    """Add cyclical temporal features to preserve periodicity."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Cyclical encoding for hour (24-hour period)
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    
    # Cyclical encoding for day of week (7-day period)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    
    return df

def add_derivative_features(df, column='temp_value', window=1):
    """Add derivative-based features to capture rate of change dynamics."""
    df = df.copy()
    
    df['velocity'] = df[column].diff(window)
    df['acceleration'] = df['velocity'].diff(window)
    df['energy'] = df[column]**2
    df['roll_std'] = df[column].rolling(window=144).std()
    
    # Fill NaN values created by diff and rolling operations
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

def add_volatility_features(df, column='temp_value', window=10):
    """Add volatility-based features to detect flat-line sensor malfunctions."""
    df = df.copy()
    rolling_std = df[column].rolling(window=10).std()
    rolling_std.fillna(method='bfill', inplace=True)
    
    epsilon = 1e-3
    df['static_sensor_alert'] = 1.0 / (rolling_std + epsilon)
    
    # Log transform to compress extreme values
    df['static_sensor_alert'] = np.log1p(df['static_sensor_alert'])

    return df

def add_statistical_features(sequences):
    """Add statistical features computed over each sequence window."""
    n_samples = sequences.shape[0]
    seq_len = sequences.shape[1]
    n_base_features = sequences.shape[2]
    
    stats_features = []
    
    for seq in sequences:
        # Calculate stats only on the first feature (temp_value)
        temp_values = seq[:, 0]
        
        mean_val = np.mean(temp_values)
        std_val = np.std(temp_values)
        range_val = np.max(temp_values) - np.min(temp_values)
        
        # Repeat stats for each timestep in the sequence
        stats = np.column_stack([
            np.full(seq_len, mean_val),
            np.full(seq_len, std_val),
            np.full(seq_len, range_val)
        ])
        
        stats_features.append(stats)
    
    stats_features = np.array(stats_features)
    
    # Concatenate with original sequences
    enhanced_sequences = np.concatenate([sequences, stats_features], axis=2)
    
    return enhanced_sequences

def prepare_data_with_features(df, sequence_length, feature_mode='all'):
    """
    Main data preparation pipeline with full feature engineering.
    
    Creates 13-feature representation:
        1. temp_value (raw temperature)
        2-5. hour_sin, hour_cos, dow_sin, dow_cos (temporal context)
        6-10. energy, velocity, acceleration, roll_std, static_sensor_alert (dynamics)
        11-13. mean, std, range (sequence statistics)
    """
    logger.info(f"Preparing data with feature_mode='{feature_mode}'")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle missing values
    if df['temp_value'].isna().any():
        logger.warning("Missing values detected - filling with forward fill")
        df['temp_value'].fillna(method='ffill', inplace=True)
    
    if feature_mode == 'all':
        # Apply all feature engineering steps
        df = add_temporal_features(df)
        df = add_derivative_features(df)
        df = add_volatility_features(df)
        
        feature_cols = [
            'temp_value', 
            'hour_sin', 'hour_cos', 
            'dow_sin', 'dow_cos',
            'energy', 'velocity', 'acceleration', 'roll_std', 'static_sensor_alert'
        ]
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = feature_cols + ['mean', 'std', 'range']
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    logger.info(f"Sequences shape: {sequences.shape}")
    logger.info(f"Features ({sequences.shape[2]}): {feature_names}")
    
    return sequences, feature_names

def validate_expanding_window_splits(data, splits, sequence_length):
    """Validate that splits don't have leakage and are properly ordered"""
    logger.info("Validating expanding window splits...")
    
    for i, (train_chunk, val_chunk) in enumerate(splits):
        # Check sizes
        assert len(train_chunk) > sequence_length, f"Fold {i+1}: train too small"
        assert len(val_chunk) > sequence_length, f"Fold {i+1}: val too small"
        
        # Check expanding property
        if i > 0:
            prev_train_size = len(splits[i-1][0])
            assert len(train_chunk) > prev_train_size, f"Fold {i+1}: not expanding"
        
        # Check no overlap (simplified check)
        assert len(train_chunk) + len(val_chunk) <= len(data), f"Fold {i+1}: data overlap"
    
    logger.info("✓ All splits validated successfully")

def create_expanding_window_splits(
    data, sequence_length, n_folds=5, val_ratio=0.20, use_purge=True, purge_mult=2
):
    """Create expanding window splits with enhanced purge gap - returns DataFrames with real timestamps"""
    data_sorted = data.sort_values('timestamp').reset_index(drop=True)
    N = len(data_sorted)
    if N <= sequence_length:
        raise ValueError(f"Data too small. Need at least {sequence_length + 1} points.")

    # Enhanced purge gap (default 2x sequence_length for autocorrelated data)
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

    # Validate splits
    validate_expanding_window_splits(data_sorted, splits, sequence_length)
    
    return splits

# ============================================================================
# MLP AUTOENCODER MODEL CLASS
# ============================================================================

class MLPAutoencoder:
    """Multi-Layer Perceptron (MLP) Autoencoder for time series anomaly detection."""
    
    def __init__(self, sequence_length, n_features,
                 hidden_units=64, encoding_dim=32, 
                 learning_rate=0.001, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the MLP Autoencoder architecture."""
        inp = Input(shape=(self.sequence_length, self.n_features))
        
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
        
        # Output - reconstruct to original shape
        x = Dense(self.sequence_length * self.n_features)(x)
        out = Reshape((self.sequence_length, self.n_features))(x)
        
        self.model = Model(inp, out)
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return self.model
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, patience=15, callbacks=None):
        """Train the autoencoder on normal data."""
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
        """Free memory by deleting model and clearing session."""
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
        self.total_time = 0.0  # To track total optimization time
    
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
                
                # Prepare data with feature engineering
                X_train, feature_names = prepare_data_with_features(
                    df_train_fold, self.sequence_length, feature_mode='all'
                )
                X_val, _ = prepare_data_with_features(
                    df_val_fold, self.sequence_length, feature_mode='all'
                )
                
                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"    No sequences created - skipping")
                    continue
                
                n_features = X_train.shape[2]  # Should be 13
                
                # Scale each feature independently
                scalers = []
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                
                for i in range(n_features):
                    scaler = StandardScaler()
                    X_train_scaled[:, :, i] = scaler.fit_transform(
                        X_train[:, :, i].reshape(-1, 1)
                    ).reshape(X_train.shape[0], self.sequence_length)
                    X_val_scaled[:, :, i] = scaler.transform(
                        X_val[:, :, i].reshape(-1, 1)
                    ).reshape(X_val.shape[0], self.sequence_length)
                    scalers.append(scaler)
                
                # Build model
                autoencoder = MLPAutoencoder(
                    sequence_length=self.sequence_length,
                    n_features=n_features,
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
        Returns total time spent on optimization.
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
        self.total_time = duration.total_seconds()
        
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
        logger.info(f"Total seconds: {self.total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_loss:.6f}")
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Trials: {completed_trials} completed, {pruned_trials} pruned")
        logger.info(f"{'='*80}\n")
        
        return {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'study': self.study,
            'all_results': self.all_results,
            'duration': self.total_time,  # Return total time in seconds
            'duration_formatted': str(duration),
            'trial_stats': {
                'total': len(self.study.trials),
                'completed': completed_trials,
                'pruned': pruned_trials
            }
        }

# ============================================================================
# VISUALIZATION AND SAVING FUNCTIONS
# ============================================================================

def save_bohb_results(results: Dict[str, Any], output_dir: str = '../HPO Results/Representative/mlp_bohb_results'):
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
        'duration_formatted': results['duration_formatted'],
        'hyperband_config': {
            'min_epochs': 20,
            'max_epochs': 100,
            'eta': 3,
            'n_trials': 50
        },
        'timestamp': datetime.now().isoformat(),
        'feature_engineering': 'full_13_features',
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
        'importance_file': importance_file,
        'duration_seconds': results['duration']
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
    Returns total time spent on optimization.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("="*80)
    
    if csv_path is None:
        csv_path = CSV_PATH
    
    if not os.path.exists(csv_path):
        logger.error(f"ERROR: '{csv_path}' not found.")
        return None
    
    # Load data
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for missing values
    if df['temp_value'].isna().any():
        logger.warning("Missing values detected - filling with forward fill")
        df['temp_value'].fillna(method='ffill', inplace=True)
    
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
    duration_seconds = results['duration']
    
    logger.info(f"Total optimization time: {results['duration_formatted']}")
    logger.info(f"Total optimization seconds: {duration_seconds:.2f}s")
    
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
    logger.info(f"Number of features: 13 (with feature engineering)")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Total optimization time: {results['duration_formatted']}")
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
            'eta': eta,
            'total_time_seconds': duration_seconds
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
        # Configuration
        SEQUENCE_LENGTH = 24
        N_TRIALS = 60
        N_FOLDS = 5
        MIN_EPOCHS = 30
        MAX_EPOCHS = 100
        ETA = 2
        RANDOM_STATE = 42
        
        if not os.path.exists(CSV_PATH):
            logger.error(f"ERROR: '{CSV_PATH}' not found.")
            logger.info("Please ensure the training data file exists at the specified path.")
            logger.info(f"Current working directory: {os.getcwd()}")
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
                csv_path=CSV_PATH,
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
                logger.info(f"Total optimization time: {results_dict['results']['duration_formatted']}")
                logger.info(f"Total optimization seconds: {results_dict['config']['total_time_seconds']:.2f}s")
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
                logger.info(f"  Average time per trial: {results_dict['config']['total_time_seconds']/total_trials:.2f}s")

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