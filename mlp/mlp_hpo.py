# ============================================================================
# MLP AUTOENCODER - BAYESIAN HYPERPARAMETER OPTIMIZATION
# ============================================================================
"""
Hyperparameter optimization for MLP Autoencoder using Bayesian Optimization
with expanding window cross-validation for time series anomaly detection.

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
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import json
from datetime import datetime
import joblib
import warnings
import os
import random
import matplotlib.pyplot as plt
import logging
import gc
import sys

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
"""
Set all random seeds for reproducible results across runs.
Critical for scientific publication.
"""
RANDOM_SEED = 42

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
"""
Configure logging to both file and console for experiment tracking.
"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mlp_hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CLUSTER CONFIGURATION - CHANGE ONLY THIS SECTION
# ============================================================================

CLUSTER = 'c4'  # Options: 'c1', 'c2', 'c3', 'c4'

# Automatically derived paths
CSV_PATH = f'1. Datasets/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_PREFIX = f'{CLUSTER}_mlp_hpo_results'

logger.info(f"="*80)
logger.info(f"Configured for CLUSTER: {CLUSTER}")
logger.info(f"Data path: {CSV_PATH}")
logger.info(f"Output prefix: {OUTPUT_PREFIX}")
logger.info(f"="*80)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
"""
Enable GPU memory growth to avoid allocating all GPU memory at once.
Allows multiple experiments to run if needed.
"""
gpus = tf.config.list_physical_devices('GPU')
# Specify which GPU to use (0 or 1)
TARGET_GPU = 1  # Change to 1 for RTX 4070 Ti SUPER

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth first
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Restrict to specific GPU
        tf.config.set_visible_devices(gpus[TARGET_GPU], 'GPU')
        
        logger.info(f"✓ Using GPU {TARGET_GPU}: {gpus[TARGET_GPU].name}")
        logger.info(f"  Other GPUs hidden for consistent benchmarking")
        
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
tf.config.experimental.enable_op_determinism()
# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger.info("="*80)
logger.info("MLP AUTOENCODER - BAYESIAN HYPERPARAMETER OPTIMIZATION")
logger.info("="*80)
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Random seed: {RANDOM_SEED}")
logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_sequences(data, sequence_length):
    """
    Create sliding window sequences preserving chronological order.
    
    Args:
        data: numpy array of shape (n_timesteps, n_features)
        sequence_length: length of each sequence window
        
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
    """
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


def add_temporal_features(df):
    """
    Add cyclical temporal features to preserve periodicity.
    Uses sine/cosine encoding to avoid discontinuity (e.g., hour 23 -> 0).
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        df: DataFrame with added temporal features
    """
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
    """
    Add derivative-based features to capture rate of change dynamics.
    
    Features:
        - velocity: first derivative (rate of change)
        - acceleration: second derivative (rate of change of rate of change)
        - energy: squared values
        - roll_std: rolling standard deviation (144 timesteps = 24 hours)
    
    Args:
        df: DataFrame with temperature column
        column: name of the temperature column
        window: window size for diff calculation
        
    Returns:
        df: DataFrame with added derivative features
    """
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
    """
    Add volatility-based features to detect flat-line sensor malfunctions.
    
    The static_sensor_alert feature uses inverse of rolling std:
        - High values indicate low volatility (potential sensor malfunction)
        - Low values indicate normal sensor behavior
    
    Args:
        df: DataFrame with temperature column
        column: name of the temperature column
        window: window size for rolling std calculation
        
    Returns:
        df: DataFrame with added volatility features
    """
    df = df.copy()
    rolling_std = df[column].rolling(window=10).std()
    rolling_std.fillna(method='bfill', inplace=True)
    
    epsilon = 1e-3  # Prevent division by zero
    df['static_sensor_alert'] = 1.0 / (rolling_std + epsilon)
    
    # Log transform to compress extreme values
    df['static_sensor_alert'] = np.log1p(df['static_sensor_alert'])

    return df


def add_statistical_features(sequences):
    """
    Add statistical features computed over each sequence window.
    
    For each sequence, calculates aggregate statistics and repeats them
    across all timesteps. This provides sequence-level context.
    
    Features:
        - mean: average temperature in the window
        - std: standard deviation in the window
        - range: max - min in the window
    
    Args:
        sequences: numpy array of shape (n_samples, sequence_length, n_base_features)
        
    Returns:
        enhanced_sequences: array of shape (n_samples, sequence_length, n_base_features + 3)
    """
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
    
    Args:
        df: DataFrame with 'timestamp' and 'temp_value' columns
        sequence_length: length of sliding window (e.g., 24 = 4 hours)
        feature_mode: 'all' uses full 13-feature pipeline
        
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, 13)
        feature_names: list of feature names
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
    """
    Multi-Layer Perceptron (MLP) Autoencoder for time series anomaly detection.
    
    Architecture:
        Input (sequence_length × n_features) 
        → Flatten 
        → Dense(hidden_units*2) → Dropout 
        → Dense(hidden_units) → Dropout 
        → Dense(encoding_dim) [bottleneck]
        → Dense(hidden_units) → Dropout 
        → Dense(hidden_units*2) → Dropout 
        → Reshape 
        → Output (sequence_length × n_features)
    
    The bottleneck forces the model to learn compressed representations,
    enabling anomaly detection via reconstruction error.
    """
    
    def __init__(self, sequence_length, n_features,
                 hidden_units=64, encoding_dim=32, 
                 learning_rate=0.001, dropout_rate=0.2):
        """
        Initialize MLP Autoencoder.
        
        Args:
            sequence_length: length of input sequences (e.g., 24)
            n_features: number of features per timestep (e.g., 13)
            hidden_units: number of units in hidden layers
            encoding_dim: size of bottleneck (compressed representation)
            learning_rate: learning rate for Adam optimizer
            dropout_rate: dropout rate for regularization
        """
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
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, patience=15, verbose=0):
        """
        Train the autoencoder on normal data.
        
        Args:
            X_train: training sequences (n_samples, sequence_length, n_features)
            X_val: validation sequences
            epochs: maximum number of training epochs
            batch_size: batch size for training
            patience: early stopping patience
            verbose: verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            history: training history object
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                verbose=1 if verbose > 0 else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=max(5, patience//3), 
                min_lr=1e-7,
                verbose=1 if verbose > 0 else 0
            )
        ]
        
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = output
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Preserve chronological order
        )
        
        return self.history
    
    def predict(self, X):
        """Reconstruct input sequences."""
        return self.model.predict(X, verbose=0)
    
    def get_reconstruction_errors(self, X):
        """
        Calculate reconstruction errors (MSE per sample).
        
        Args:
            X: sequences to reconstruct
            
        Returns:
            errors: array of MSE values, one per sample
        """
        preds = self.predict(X)
        return np.mean(np.square(X - preds), axis=(1, 2))
    
    def cleanup(self):
        """Free memory by deleting model and clearing session."""
        if self.model is not None:
            del self.model
            self.model = None
        tf.keras.backend.clear_session()
        gc.collect()


# ============================================================================
# HYPERPARAMETER OPTIMIZER CLASS
# ============================================================================

class HyperparameterOptimizer:
    """
    Bayesian Hyperparameter Optimization for MLP Autoencoder.
    
    Uses Gaussian Process optimization to efficiently search the hyperparameter
    space by learning from previous evaluations.
    
    Optimizes: hidden_units, encoding_dim, learning_rate, dropout_rate, batch_size
    Objective: Minimize average validation reconstruction error across CV folds
    """
    
    def __init__(self, data_splits, sequence_length, n_calls=35, random_state=42):
        """
        Initialize optimizer.
        
        Args:
            data_splits: list of (train_data, val_data) tuples from CV
            sequence_length: length of sequences (e.g., 24)
            n_calls: number of hyperparameter combinations to evaluate
            random_state: random seed for reproducibility
        """
        self.data_splits = data_splits
        self.sequence_length = sequence_length
        self.n_calls = n_calls
        self.random_state = random_state
        self.results = []
        self.best_loss = float('inf')
        self.best_params = None
        
    def define_search_space(self):
        """
        Define hyperparameter search space.
        
        Returns:
            space: list of hyperparameter dimensions for skopt
        """
        return [
            Integer(128, 512, name='hidden_units'),           # Hidden layer size
            Integer(16, 96, name='encoding_dim'),            # Bottleneck size
            Real(1e-4, 1e-3, prior='log-uniform', name='learning_rate'),  # Learning rate
            Real(0.1, 0.3, name='dropout_rate'),             # Dropout rate
            Categorical([32, 64], name='batch_size')    # Batch size
        ]
    
    # def objective_function(self, hidden_units, encoding_dim, learning_rate, 
    #                       dropout_rate, batch_size):
    #     """
    #     Objective function to minimize: average validation loss across CV folds.
        
    #     This function:
    #     1. Trains a model on each CV fold with given hyperparameters
    #     2. Calculates validation reconstruction error for each fold
    #     3. Returns average validation error across all folds
        
    #     Args:
    #         Hyperparameters to evaluate
            
    #     Returns:
    #         avg_val_loss: average validation MSE across all folds
    #     """
    #     params = {
    #         'hidden_units': int(hidden_units),
    #         'encoding_dim': int(encoding_dim),
    #         'learning_rate': float(learning_rate),
    #         'dropout_rate': float(dropout_rate),
    #         'batch_size': int(batch_size)
    #     }
        
    #     logger.info(f"\nTesting hyperparameters: {params}")
        
    #     fold_val_losses = []
    #     fold_train_times = []
        
    #     try:
    #         for fold_idx, (train_data_raw, val_data_raw) in enumerate(self.data_splits):
    #             logger.info(f"  Fold {fold_idx + 1}/{len(self.data_splits)}")
                
    #             fold_start = datetime.now()
                
    #             # Check if fold has sufficient data
    #             if len(train_data_raw) <= self.sequence_length or len(val_data_raw) <= self.sequence_length:
    #                 logger.warning(f"    Fold {fold_idx + 1}: insufficient data - skipping")
    #                 continue
                
    #             # Prepare data with full 13-feature engineering
    #             # Create temporary DataFrames for feature engineering
    #             train_timestamps = pd.date_range(start='2020-01-01', periods=len(train_data_raw), freq='10T')
    #             val_timestamps = pd.date_range(start='2020-01-01', periods=len(val_data_raw), freq='10T')
                
    #             df_train_fold = pd.DataFrame({
    #                 'timestamp': train_timestamps,
    #                 'temp_value': train_data_raw
    #             })
    #             df_val_fold = pd.DataFrame({
    #                 'timestamp': val_timestamps,
    #                 'temp_value': val_data_raw
    #             })
                
    #             # Apply full feature engineering pipeline
    #             X_train, feature_names = prepare_data_with_features(
    #                 df_train_fold, self.sequence_length, feature_mode='all'
    #             )
    #             X_val, _ = prepare_data_with_features(
    #                 df_val_fold, self.sequence_length, feature_mode='all'
    #             )
                
    #             if len(X_train) == 0 or len(X_val) == 0:
    #                 logger.warning(f"    Fold {fold_idx + 1}: no sequences created - skipping")
    #                 continue
                
    #             n_features = X_train.shape[2]  # Should be 13
                
    #             # Scale per-fold (no data leakage) - scale each feature independently
    #             scalers = []
    #             X_train_scaled = X_train.copy()
    #             X_val_scaled = X_val.copy()
                
    #             for i in range(n_features):
    #                 scaler = StandardScaler()
    #                 X_train_scaled[:, :, i] = scaler.fit_transform(
    #                     X_train[:, :, i].reshape(-1, 1)
    #                 ).reshape(X_train.shape[0], self.sequence_length)
    #                 X_val_scaled[:, :, i] = scaler.transform(
    #                     X_val[:, :, i].reshape(-1, 1)
    #                 ).reshape(X_val.shape[0], self.sequence_length)
    #                 scalers.append(scaler)
                
    #             # Build and train model
    #             autoencoder = MLPAutoencoder(
    #                 sequence_length=self.sequence_length,
    #                 n_features=n_features,  # Use actual number of features (13)
    #                 hidden_units=params['hidden_units'],
    #                 encoding_dim=params['encoding_dim'],
    #                 learning_rate=params['learning_rate'],
    #                 dropout_rate=params['dropout_rate']
    #             )
                
    #             history = autoencoder.train(
    #                 X_train_scaled, X_val_scaled,
    #                 epochs=100,
    #                 batch_size=params['batch_size'],
    #                 patience=15,
    #                 verbose=0
    #             )
                
    #             # Validate training results
    #             val_losses = history.history['val_loss']
    #             if not val_losses or all(np.isnan(v) or np.isinf(v) for v in val_losses):
    #                 logger.warning(f"    Fold {fold_idx + 1}: invalid losses (NaN/Inf) - skipping")
    #                 autoencoder.cleanup()
    #                 continue
                
    #             valid_losses = [v for v in val_losses if not (np.isnan(v) or np.isinf(v))]
    #             if not valid_losses:
    #                 logger.warning(f"    Fold {fold_idx + 1}: no valid losses - skipping")
    #                 autoencoder.cleanup()
    #                 continue
                
    #             val_loss_fold = min(valid_losses)
    #             fold_val_losses.append(val_loss_fold)
                
    #             fold_time = (datetime.now() - fold_start).total_seconds()
    #             fold_train_times.append(fold_time)
                
    #             logger.info(f"    Train: {X_train_scaled.shape[0]} seqs, Val: {X_val_scaled.shape[0]} seqs")
    #             logger.info(f"    Val Loss: {val_loss_fold:.6f}, Time: {fold_time:.1f}s")
                
    #             # Cleanup
    #             autoencoder.cleanup()
    #             del X_train, X_val, X_train_scaled, X_val_scaled, scalers
    #             gc.collect()
            
    #         # Calculate average validation loss
    #         if not fold_val_losses:
    #             logger.warning("  No valid folds completed - returning penalty")
    #             return 1e10
            
    #         avg_val_loss = np.mean(fold_val_losses)
    #         avg_train_time = np.mean(fold_train_times)
            
    #         logger.info(f"  Average Val Loss: {avg_val_loss:.6f}")
    #         logger.info(f"  Average Train Time: {avg_train_time:.1f}s")
            
    #         # Store results
    #         self.results.append({
    #             'params': params,
    #             'avg_val_loss': avg_val_loss,
    #             'fold_losses': fold_val_losses,
    #             'avg_train_time': avg_train_time,
    #             'n_folds_completed': len(fold_val_losses)
    #         })
            
    #         # Update best
    #         if avg_val_loss < self.best_loss:
    #             self.best_loss = avg_val_loss
    #             self.best_params = params.copy()
    #             logger.info(f"  ✓ NEW BEST! Loss: {avg_val_loss:.6f}")
            
    #         return avg_val_loss
            
    #     except tf.errors.ResourceExhaustedError as e:
    #         logger.error(f"  GPU OOM Error: {str(e)}")
    #         return 1e10
    #     except Exception as e:
    #         logger.error(f"  Unexpected error: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         return 1e10
    #     finally:
    #         # Final cleanup
    #         tf.keras.backend.clear_session()
    #         gc.collect()
            
    def objective_function(self, hidden_units, encoding_dim, learning_rate, 
                          dropout_rate, batch_size):
        """
        Objective function to minimize: average validation loss across CV folds.
        
        This function:
        1. Trains a model on each CV fold with given hyperparameters
        2. Calculates validation reconstruction error for each fold
        3. Returns average validation error across all folds
        
        Args:
            Hyperparameters to evaluate
            
        Returns:
            avg_val_loss: average validation MSE across all folds
        """
        params = {
            'hidden_units': int(hidden_units),
            'encoding_dim': int(encoding_dim),
            'learning_rate': float(learning_rate),
            'dropout_rate': float(dropout_rate),
            'batch_size': int(batch_size)
        }
        
        logger.info(f"\nTesting hyperparameters: {params}")
        
        fold_val_losses = []
        fold_train_times = []
        
        try:
            for fold_idx, (df_train_fold, df_val_fold) in enumerate(self.data_splits):
                logger.info(f"  Fold {fold_idx + 1}/{len(self.data_splits)}")
                
                fold_start = datetime.now()
                
                # Check if fold has sufficient data
                if len(df_train_fold) <= self.sequence_length or len(df_val_fold) <= self.sequence_length:
                    logger.warning(f"    Fold {fold_idx + 1}: insufficient data - skipping")
                    continue
                
                # Apply full feature engineering pipeline
                X_train, feature_names = prepare_data_with_features(
                    df_train_fold, self.sequence_length, feature_mode='all'
                )
                X_val, _ = prepare_data_with_features(
                    df_val_fold, self.sequence_length, feature_mode='all'
                )
                
                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"    Fold {fold_idx + 1}: no sequences created - skipping")
                    continue
                
                n_features = X_train.shape[2]  # Should be 13
                
                # Scale per-fold (no data leakage) - scale each feature independently
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
                
                # Build and train model
                autoencoder = MLPAutoencoder(
                    sequence_length=self.sequence_length,
                    n_features=n_features,  # Use actual number of features (13)
                    hidden_units=params['hidden_units'],
                    encoding_dim=params['encoding_dim'],
                    learning_rate=params['learning_rate'],
                    dropout_rate=params['dropout_rate']
                )
                
                history = autoencoder.train(
                    X_train_scaled, X_val_scaled,
                    epochs=100,
                    batch_size=params['batch_size'],
                    patience=15,
                    verbose=0
                )
                
                # Validate training results
                val_losses = history.history['val_loss']
                if not val_losses or all(np.isnan(v) or np.isinf(v) for v in val_losses):
                    logger.warning(f"    Fold {fold_idx + 1}: invalid losses (NaN/Inf) - skipping")
                    autoencoder.cleanup()
                    continue
                
                valid_losses = [v for v in val_losses if not (np.isnan(v) or np.isinf(v))]
                if not valid_losses:
                    logger.warning(f"    Fold {fold_idx + 1}: no valid losses - skipping")
                    autoencoder.cleanup()
                    continue
                
                val_loss_fold = min(valid_losses)
                fold_val_losses.append(val_loss_fold)
                
                fold_time = (datetime.now() - fold_start).total_seconds()
                fold_train_times.append(fold_time)
                
                logger.info(f"    Train: {X_train_scaled.shape[0]} seqs, Val: {X_val_scaled.shape[0]} seqs")
                logger.info(f"    Val Loss: {val_loss_fold:.6f}, Time: {fold_time:.1f}s")
                
                # Cleanup
                autoencoder.cleanup()
                del X_train, X_val, X_train_scaled, X_val_scaled, scalers
                gc.collect()
            
            # Calculate average validation loss
            if not fold_val_losses:
                logger.warning("  No valid folds completed - returning penalty")
                return 1e10
            
            avg_val_loss = np.mean(fold_val_losses)
            avg_train_time = np.mean(fold_train_times)
            
            logger.info(f"  Average Val Loss: {avg_val_loss:.6f}")
            logger.info(f"  Average Train Time: {avg_train_time:.1f}s")
            
            # Store results
            self.results.append({
                'params': params,
                'avg_val_loss': avg_val_loss,
                'fold_losses': fold_val_losses,
                'avg_train_time': avg_train_time,
                'n_folds_completed': len(fold_val_losses)
            })
            
            # Update best
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_params = params.copy()
                logger.info(f"  ✓ NEW BEST! Loss: {avg_val_loss:.6f}")
            
            return avg_val_loss
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"  GPU OOM Error: {str(e)}")
            return 1e10
        except Exception as e:
            logger.error(f"  Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1e10
        finally:
            # Final cleanup
            tf.keras.backend.clear_session()
            gc.collect()
    
    def run_optimization(self):
        """
        Run Bayesian optimization.
        
        Returns:
            results: dict with best_params, best_loss, optimization_result, all_results
        """
        space = self.define_search_space()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING BAYESIAN OPTIMIZATION")
        logger.info(f"{'='*80}")
        logger.info(f"Total evaluations: {self.n_calls}")
        logger.info(f"CV folds: {len(self.data_splits)}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Random seed: {self.random_state}")
        logger.info(f"{'='*80}\n")
        
        @use_named_args(space)
        def objective(**params):
            loss = self.objective_function(**params)
            
            # Handle invalid losses
            if np.isnan(loss) or np.isinf(loss) or loss >= 1e10:
                loss = 1e10
                logger.warning(f"  Invalid loss detected, using penalty: {loss}")
            
            return loss
        
        start_time = datetime.now()
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.n_calls,
            n_initial_points=10,
            acq_func='EI',
            random_state=self.random_state,
            verbose=True
        )
        
        duration = datetime.now() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {duration}")
        logger.info(f"Best validation loss: {result.fun:.6f}")
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"{'='*80}\n")
        
        return {
            'best_params': self.best_params,
            'best_loss': result.fun,
            'optimization_result': result,
            'all_results': self.results,
            'convergence_data': {
                'func_vals': result.func_vals,
                'x_iters': result.x_iters
            }
        }


# ============================================================================
# HELPER FUNCTIONS FOR SAVING AND VISUALIZATION
# ============================================================================

def save_optimization_results(results, output_dir='mlp_hpo_results_512'):
    """
    Save optimization results to JSON file with convergence plots.
    
    Args:
        results: dict from optimizer.run_optimization()
        output_dir: directory to save results
        
    Returns:
        results_file: path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_file = os.path.join(output_dir, f"c2_mlp_hpo_results_{timestamp}.json")
    results_file = os.path.join(output_dir, f"{OUTPUT_PREFIX}_{timestamp}.json")
    
    
    # Prepare JSON-serializable results
    json_results = []
    for r in results['all_results']:
        json_results.append({
            'params': r['params'],
            'avg_val_loss': float(r['avg_val_loss']) if r['avg_val_loss'] < 1e10 else None,
            'fold_losses': [float(x) for x in r.get('fold_losses', [])],
            'avg_train_time': r.get('avg_train_time', 0),
            'n_folds_completed': r.get('n_folds_completed', 0)
        })
    
    final_results = {
        'optimization_method': 'Bayesian_GP',
        'best_params': results['best_params'],
        'best_loss': float(results['best_loss']),
        'timestamp': datetime.now().isoformat(),
        'total_evaluations': len(json_results),
        'convergence_data': {
            'func_vals': [float(v) for v in results['convergence_data']['func_vals']],
            'evaluation_order': list(range(len(results['convergence_data']['func_vals'])))
        },
        'all_results': json_results,
        # In metadata.json, add:
        'environment': {
            'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__,
            'python_version': sys.version,
            'random_seed': RANDOM_SEED
}
    }
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"✓ Results saved to {results_file}")
    
    # Generate convergence plot
    plot_file = results_file.replace('.json', '_convergence.png')
    plot_convergence(results, plot_file)
    
    return results_file


def plot_convergence(results, filename):
    """
    Plot optimization convergence showing progress over evaluations.
    
    Creates two plots:
    1. Loss per evaluation with cumulative best
    2. Distribution of all losses
    
    Args:
        results: dict from optimizer.run_optimization()
        filename: path to save plot
    """
    func_vals = results['convergence_data']['func_vals']
    valid_vals = [v for v in func_vals if v < 1e10]
    
    if not valid_vals:
        logger.warning("No valid values to plot")
        return
    
    cumulative_min = np.minimum.accumulate(valid_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss per evaluation
    ax1.plot(valid_vals, 'o-', alpha=0.6, label='Loss per evaluation')
    ax1.plot(cumulative_min, 'r-', linewidth=2, label='Best loss so far')
    ax1.set_xlabel('Evaluation', fontsize=12)
    ax1.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss distribution
    ax2.hist(valid_vals, bins=30, alpha=0.7, edgecolor='black')
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

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_hyperparameter_optimization(csv_path=None,
                                    sequence_length=24,
                                    n_calls=35,
                                    n_folds=5,
                                    random_state=42):
    """
    Run complete hyperparameter optimization pipeline.
    
    Steps:
    1. Load and validate training data
    2. Create expanding window CV splits
    3. Run Bayesian optimization
    4. Save results and plots
    
    Args:
        csv_path: path to training data CSV
        sequence_length: length of sequences (e.g., 24 = 4 hours)
        n_calls: number of hyperparameter evaluations
        n_folds: number of CV folds
        random_state: random seed for reproducibility
        
    Returns:
        results_dict: dict with optimization results and file paths
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD AND VALIDATE DATA")
    logger.info("="*80)
    if csv_path is None:
        csv_path = CSV_PATH  # ← Use global constant
    
    # Check if file exists
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
    logger.info(f"  Value range: [{df['temp_value'].min():.2f}, {df['temp_value'].max():.2f}]")
    
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
            
    except Exception as e:
        logger.error(f"ERROR creating splits: {str(e)}")
        return None
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: RUN BAYESIAN OPTIMIZATION")
    logger.info("="*80)
    
    optimizer = HyperparameterOptimizer(
        data_splits=data_splits,
        sequence_length=sequence_length,
        n_calls=n_calls,
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
    
    results_file = save_optimization_results(results)
    
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Number of folds: {len(data_splits)}")
    logger.info(f"Total evaluations: {len(results['all_results'])}")
    logger.info(f"Best validation loss: {results['best_loss']:.6f}")
    logger.info(f"Best hyperparameters:")
    for k, v in results['best_params'].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Results file: {results_file}")
    logger.info("="*80)
    
    return {
        'results': results,
        'sequence_length': sequence_length,
        'files': {'results': results_file}
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("MLP AUTOENCODER - BAYESIAN HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info("  ✓ Random seed: 42 (reproducibility)")
    logger.info("="*80)
    
    try:
        # Configuration
        SEQUENCE_LENGTH = 24
        N_CALLS = 25
        N_FOLDS = 5
        RANDOM_STATE = 42
        
        # Check if data file exists
        if not os.path.exists(CSV_PATH):
            logger.error(f"ERROR: '{CSV_PATH}' not found.")
            logger.info("Please ensure the training data file exists at the specified path.")
        else:
            # ================================================================
            # STEP 1: RUN HYPERPARAMETER OPTIMIZATION
            # ================================================================
            logger.info("\n" + "="*80)
            logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
            logger.info("="*80)
            
            results_dict = run_hyperparameter_optimization(
                csv_path=CSV_PATH,
                sequence_length=SEQUENCE_LENGTH,
                n_calls=N_CALLS,
                n_folds=N_FOLDS,
                random_state=RANDOM_STATE
            )

            if results_dict is not None:
                logger.info("\n" + "="*80)
                logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY!")
                logger.info("="*80)
                logger.info(f"Best validation loss: {results_dict['results']['best_loss']:.6f}")
                logger.info(f"Best parameters:")
                for k, v in results_dict['results']['best_params'].items():
                    logger.info(f"  {k}: {v}")
                
                logger.info("\n" + "="*80)
                logger.info("NEXT STEPS")
                logger.info("="*80)
                logger.info("1. Use best hyperparameters to train final model")
                logger.info("2. Test on labeled anomaly data")
                logger.info("3. Compare with BiLSTM/LSTM results for publication")
                logger.info("="*80)
            else:
                logger.error("Hyperparameter optimization failed. Check logs for details.")
    
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user.")
        logger.info("Progress has been logged to file.")
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("SCRIPT COMPLETED")
    logger.info("="*80)