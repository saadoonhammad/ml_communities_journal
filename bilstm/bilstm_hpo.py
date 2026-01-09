# ============================================================================
# BiLSTM AUTOENCODER - BAYESIAN HYPERPARAMETER OPTIMIZATION WITH FEATURE ENGINEERING
# ============================================================================
"""
Hyperparameter optimization for single-layer BiLSTM Autoencoder using Bayesian Optimization
with expanding window cross-validation for time series anomaly detection.

Integrates 13-feature engineering pipeline:
- Raw temperature
- Temporal features (cyclical encoding)
- Derivative features (velocity, acceleration, energy, volatility)
- Statistical features (mean, std, range)

Architecture: Single-layer Bidirectional LSTM with tanh activation
Optimizes: lstm_units, encoding_dim, learning_rate, dropout_rate, batch_size
Fixed: num_layers=1, lstm_activation='tanh', epochs=100
Objective: Minimize validation reconstruction error (MSE)
Validation: 5-fold expanding window cross-validation (chronological order preserved)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
from collections import namedtuple
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

# Define SimpleResult globally for safe pickling/unpickling
SimpleResult = namedtuple('SimpleResult', ['x_iters', 'func_vals', 'x', 'fun', 'space_dims'])

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
        logging.FileHandler(f'bilstm_fe_hpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
"""
Enable GPU memory growth and restrict to specific GPU for consistent benchmarking.
"""
TARGET_GPU = 0  # Change to 1 for second GPU if needed

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
logger.info("BiLSTM AUTOENCODER - BAYESIAN HPO WITH FEATURE ENGINEERING")
logger.info("="*80)
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Random seed: {RANDOM_SEED}")
logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

# ============================================================================
# CLUSTER CONFIGURATION - CHANGE THIS FOR DIFFERENT CLUSTERS
# ============================================================================

CLUSTER = 'c4'  # ← ONLY CHANGE THIS LINE!

# Automatically derived paths
CSV_PATH = f'1. Datasets/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_PREFIX = f'{CLUSTER}_bilstm_fe_hpo_results'

logger.info(f"Configured for CLUSTER: {CLUSTER}")
logger.info(f"Data path: {CSV_PATH}")


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

        # Return DataFrames with real timestamps instead of just values
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

    # Validate splits (now with DataFrames)
    validate_expanding_window_splits(data_sorted, splits, sequence_length)
    
    return splits


# ============================================================================
# BiLSTM AUTOENCODER MODEL CLASS
# ============================================================================

class BILSTMAutoencoder:
    """
    Bidirectional LSTM Autoencoder for time series anomaly detection.
    
    Supports 1 or 2 LSTM layers with configurable architecture.
    Uses concat merge mode (doubles effective LSTM size).
    
    Architecture (1 layer):
        Input → Bi-LSTM(concat) → Dense(encoding) → RepeatVector 
        → Bi-LSTM(concat) → TimeDistributed(Dense) → Output
    
    Architecture (2 layers):
        Input → Bi-LSTM(concat) → Bi-LSTM(concat) → Dense(encoding) 
        → RepeatVector → Bi-LSTM(concat) → Bi-LSTM(concat) 
        → TimeDistributed(Dense) → Output
    """
    
    def __init__(self, sequence_length, n_features=1,
                 encoding_dim=32,
                 num_layers=1,
                 lstm_units=50,
                 lstm_units_2=32,
                 learning_rate=0.001,
                 dropout_rate=0.2,
                 lstm_activation='tanh',
                 use_lr_warmup=True):
        """
        Initialize BiLSTM Autoencoder (single-layer configuration).
        
        Args:
            sequence_length: length of input sequences
            n_features: number of features per timestep
            encoding_dim: size of bottleneck (compressed representation)
            num_layers: fixed to 1 (single LSTM layer)
            lstm_units: units in LSTM layer
            lstm_units_2: not used (kept for compatibility)
            learning_rate: learning rate for Adam optimizer
            dropout_rate: dropout rate for regularization
            lstm_activation: activation function (fixed to 'tanh')
            use_lr_warmup: whether to use learning rate warmup
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lstm_activation = lstm_activation
        self.use_lr_warmup = use_lr_warmup
        self.model = None
        self.history = None

    def build_model(self):
        """Build the (stacked) Bi-LSTM Autoencoder with concat merge."""
        inp = Input(shape=(self.sequence_length, self.n_features))

        # Encoder (Bidirectional with concat)
        if self.num_layers == 1:
            x = Bidirectional(
                    LSTM(self.lstm_units, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=False),
                    merge_mode='concat'
                )(inp)
        else:
            x = Bidirectional(
                    LSTM(self.lstm_units, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=True),
                    merge_mode='concat'
                )(inp)
            x = Bidirectional(
                    LSTM(self.lstm_units_2, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=False),
                    merge_mode='concat'
                )(x)

        # Bottleneck
        z = Dense(self.encoding_dim, activation='tanh')(x)

        # Decoder (Bidirectional with concat)
        y = RepeatVector(self.sequence_length)(z)
        if self.num_layers == 1:
            y = Bidirectional(
                    LSTM(self.lstm_units, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=True),
                    merge_mode='concat'
                )(y)
        else:
            y = Bidirectional(
                    LSTM(self.lstm_units_2, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=True),
                    merge_mode='concat'
                )(y)
            y = Bidirectional(
                    LSTM(self.lstm_units, activation=self.lstm_activation,
                         dropout=self.dropout_rate, return_sequences=True),
                    merge_mode='concat'
                )(y)

        out = TimeDistributed(Dense(self.n_features))(y)

        self.model = Model(inp, out)
        
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return self.model

    def warmup_schedule(self, epoch, lr):
        """Learning rate warmup for first few epochs"""
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return self.learning_rate * (epoch + 1) / warmup_epochs
        return lr

    def fit(self, X_train, X_val, epochs=100, batch_size=32, verbose=0, patience=15):
        """
        Train the autoencoder on normal data.
        
        Args:
            X_train: training sequences (n_samples, sequence_length, n_features)
            X_val: validation sequences
            epochs: maximum number of training epochs
            batch_size: batch size for training
            verbose: verbosity level (0=silent, 1=progress bar)
            patience: early stopping patience
            
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
        
        if self.use_lr_warmup:
            callbacks.append(LearningRateScheduler(self.warmup_schedule, verbose=0))

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
        """Properly cleanup model and free memory"""
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
    Bayesian Hyperparameter Optimization for Single-Layer BiLSTM Autoencoder with Feature Engineering.
    
    Uses Gaussian Process optimization to efficiently search the hyperparameter
    space by learning from previous evaluations.
    
    Optimizes: lstm_units, encoding_dim, learning_rate, dropout_rate, batch_size
    Fixed: num_layers=1, lstm_activation='tanh', epochs=100
    Objective: Minimize average validation reconstruction error across CV folds
    """
    
    def __init__(self, data_splits, sequence_length, scaler_type='standard',
                 n_calls=50, random_state=42):
        """
        Initialize optimizer.
        
        Args:
            data_splits: list of (train_data, val_data) tuples from CV
            sequence_length: length of sequences (e.g., 24)
            scaler_type: 'standard' or 'minmax'
            n_calls: number of hyperparameter combinations to evaluate
            random_state: random seed for reproducibility
        """
        self.data_splits = data_splits
        self.sequence_length = sequence_length
        self.scaler_type = scaler_type
        self.n_calls = n_calls
        self.random_state = random_state
        self.results = []
        self.best_loss = float('inf')
        self.best_params = None

    def define_search_space(self):
        """
        Define hyperparameter search space for single-layer BiLSTM.
        
        Returns:
            space: list of hyperparameter dimensions for skopt
        """
        return [
            Integer(64, 512, name='lstm_units'),
            Integer(16, 128, name='encoding_dim'),
            Real(5e-5, 1e-3, prior='log-uniform', name='learning_rate'),
            Real(0.1, 0.4, name='dropout_rate'),
            Categorical([32, 64], name='batch_size')
        ]
    
    def _get_checkpoint_filename(self):
        """Standardized checkpoint filename with all relevant parameters"""
        return (f"bilstm_fe_checkpoint_s{self.sequence_length}_"
                f"f{len(self.data_splits)}_"
                f"c{self.n_calls}_"
                f"r{self.random_state}_"
                f"{self.scaler_type}.pkl")

    def objective_function(self, lstm_units, encoding_dim, learning_rate, 
                           dropout_rate, batch_size):
        """
        Objective function to minimize: average validation loss across CV folds.
        
        This function:
        1. For each CV fold, creates temp DataFrame for feature engineering
        2. Applies full 13-feature pipeline
        3. Trains single-layer BiLSTM model with given hyperparameters
        4. Calculates validation reconstruction error for each fold
        5. Returns average validation error across all folds
        
        Args:
            Hyperparameters to evaluate
            
        Returns:
            avg_val_loss: average validation MSE across all folds
        """
        params = {
            'lstm_units': int(lstm_units),
            'encoding_dim': int(encoding_dim),
            'learning_rate': float(learning_rate),
            'dropout_rate': float(dropout_rate),
            'batch_size': int(batch_size)
        }
        
        logger.info(f"Testing: {params}")
        
        fold_val_losses = []
        fold_train_times = []
        error_type = None
        
        try:
            for fold, (train_data_df, val_data_df) in enumerate(self.data_splits):
                logger.info(f"  Fold {fold + 1}/{len(self.data_splits)}")
                
                fold_start = datetime.now()

                if len(train_data_df) <= self.sequence_length or len(val_data_df) <= self.sequence_length:
                    logger.warning(f"    Skipping fold {fold + 1}: insufficient data")
                    continue

                # Use real DataFrames with real timestamps - no synthetic timestamps needed!
                X_train, feature_names = prepare_data_with_features(
                    train_data_df, self.sequence_length, feature_mode='all'
                )
                X_val, _ = prepare_data_with_features(
                    val_data_df, self.sequence_length, feature_mode='all'
                )

                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"    Skipping fold {fold + 1}: no sequences created")
                    continue

                n_features = X_train.shape[2]  # Should be 13

                # Scale per-fold (no data leakage) - scale each feature independently
                scaler = StandardScaler() if self.scaler_type == 'standard' else MinMaxScaler()
                
                X_train_scaled = X_train.copy()
                X_val_scaled = X_val.copy()
                
                for i in range(n_features):
                    X_train_scaled[:, :, i] = scaler.fit_transform(
                        X_train[:, :, i].reshape(-1, 1)
                    ).reshape(X_train.shape[0], self.sequence_length)
                    X_val_scaled[:, :, i] = scaler.transform(
                        X_val[:, :, i].reshape(-1, 1)
                    ).reshape(X_val.shape[0], self.sequence_length)

                # Build and train model (single-layer BiLSTM, tanh activation)
                autoencoder = BILSTMAutoencoder(
                    sequence_length=self.sequence_length,
                    n_features=n_features,  # Use actual number of features (13)
                    num_layers=1,  # Fixed: single layer
                    lstm_units=params['lstm_units'],
                    lstm_units_2=32,  # Not used for single layer
                    encoding_dim=params['encoding_dim'],
                    lstm_activation='tanh',  # Fixed activation
                    learning_rate=params['learning_rate'],
                    dropout_rate=params['dropout_rate'],
                    use_lr_warmup=True
                )

                # Fixed epochs with dynamic patience
                epochs = 100
                patience = max(10, epochs // 5)
                
                history = autoencoder.fit(
                    X_train_scaled, X_val_scaled,
                    epochs=epochs,
                    batch_size=params['batch_size'],
                    patience=patience,
                    verbose=0
                )
                
                # Validate training results
                val_losses = history.history['val_loss']
                if not val_losses or all(np.isnan(v) or np.isinf(v) for v in val_losses):
                    logger.warning(f"    Fold {fold + 1}: Invalid losses (NaN/Inf) - SKIPPING")
                    error_type = 'training_failure'
                    autoencoder.cleanup()
                    continue
                
                valid_losses = [v for v in val_losses if not (np.isnan(v) or np.isinf(v))]
                if not valid_losses:
                    logger.warning(f"    Fold {fold + 1}: No valid losses - SKIPPING")
                    error_type = 'training_failure'
                    autoencoder.cleanup()
                    continue
                    
                val_loss_fold = min(valid_losses)
                fold_val_losses.append(val_loss_fold)
                
                fold_time = (datetime.now() - fold_start).total_seconds()
                fold_train_times.append(fold_time)
                
                logger.info(f"    Train seq: {X_train_scaled.shape[0]}, Val seq: {X_val_scaled.shape[0]}")
                logger.info(f"    Val Loss: {val_loss_fold:.6f}, Time: {fold_time:.1f}s")
                
                # Cleanup
                autoencoder.cleanup()
                del X_train, X_val, X_train_scaled, X_val_scaled, scaler
                gc.collect()

            if not fold_val_losses:
                logger.warning("  No valid folds completed - returning penalty")
                return 1e10

            avg_val_loss = np.mean(fold_val_losses)
            avg_train_time = np.mean(fold_train_times)
            
            logger.info(f"  Average Val Loss: {avg_val_loss:.6f}")
            logger.info(f"  Average Train Time: {avg_train_time:.1f}s")

            # Store results with metadata
            self.results.append({
                'params': params,
                'avg_val_loss': avg_val_loss,
                'fold_losses': fold_val_losses,
                'avg_train_time': avg_train_time,
                'n_folds_completed': len(fold_val_losses),
                'error_type': error_type
            })

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

    def bayesian_optimization(self):
        """Run Bayesian optimization with checkpointing"""
        space = self.define_search_space()
        checkpoint_file = self._get_checkpoint_filename()
        
        DEFAULT_N_INITIAL_POINTS = 10
        
        # Checkpoint loading
        x0, y0 = None, None
        n_initial_points_new = DEFAULT_N_INITIAL_POINTS
        res_loaded = None
        
        if os.path.exists(checkpoint_file):
            logger.info(f"Checkpoint found: {checkpoint_file}. Resuming...")
            try:
                res_loaded = joblib.load(checkpoint_file)
                
                # Validate dimensions
                if hasattr(res_loaded, 'space_dims'):
                    if res_loaded.space_dims != len(space):
                        logger.warning(f"Dimension mismatch: checkpoint={res_loaded.space_dims}, space={len(space)}")
                        logger.info("Starting fresh optimization.")
                        x0, y0 = None, None
                    else:
                        # Filter valid values
                        x0, y0 = [], []
                        for x_iter, f_val in zip(res_loaded.x_iters, res_loaded.func_vals):
                            if not (np.isnan(f_val) or np.isinf(f_val) or f_val >= 1e10):
                                x0.append(x_iter)
                                y0.append(f_val)
                        
                        if len(x0) == 0:
                            logger.warning("All checkpoint values invalid. Starting fresh.")
                            x0, y0 = None, None
                        else:
                            n_initial_points_new = max(0, DEFAULT_N_INITIAL_POINTS - len(x0))
                            logger.info(f"Loaded {len(x0)} valid evaluations")
                            
                            # Restore results
                            for x_iter, f_val in zip(x0, y0):
                                params_dict = dict(zip([dim.name for dim in space], x_iter))
                                self.results.append({
                                    'params': params_dict,
                                    'avg_val_loss': f_val,
                                    'fold_losses': [],
                                    'n_folds_completed': 0
                                })
                            
                            # Update best
                            best_idx = np.argmin(y0)
                            self.best_loss = y0[best_idx]
                            self.best_params = dict(zip([dim.name for dim in space], x0[best_idx]))
                            logger.info(f"Restored best loss: {self.best_loss:.6f}")
                else:
                    # Old checkpoint format without space_dims
                    logger.warning("Old checkpoint format detected. Starting fresh.")
                    x0, y0 = None, None

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}. Starting fresh.")
                x0, y0 = None, None
        else:
            logger.info("No checkpoint found. Starting fresh optimization.")
        
        # Calculate remaining calls
        n_calls_run = len(x0) if x0 else 0
        n_calls_remaining = self.n_calls - n_calls_run
        
        if n_calls_remaining <= 0:
            logger.info("Optimization already complete!")
            return {
                'best_params': self.best_params,
                'best_loss': self.best_loss,
                'optimization_result': res_loaded,
                'all_results': self.results
            }

        logger.info(f"Starting optimization: {n_calls_remaining} remaining calls")
        logger.info(f"Initial points: {n_initial_points_new}, Folds: {len(self.data_splits)}")
        logger.info("="*80)

        @use_named_args(space)
        def objective(**params):
            loss = self.objective_function(**params)

            # Handle invalid losses
            if np.isnan(loss) or np.isinf(loss) or loss >= 1e10:
                loss = 1e10
                logger.warning(f"  Invalid loss detected, using penalty: {loss}")

            # Save checkpoint
            x_iters = [list(r['params'].values()) for r in self.results]
            func_vals = [r['avg_val_loss'] if r['avg_val_loss'] < 1e10 else 1e10 
                        for r in self.results]

            if func_vals:
                best_idx = np.argmin(func_vals)
                checkpoint_result = SimpleResult(
                    x_iters=x_iters,
                    func_vals=func_vals,
                    x=x_iters[best_idx],
                    fun=func_vals[best_idx],
                    space_dims=len(space)
                )
                
                joblib.dump(checkpoint_result, checkpoint_file)
                logger.info(f"  Checkpoint saved ({len(x_iters)} evaluations)")
            
            return loss

        start_time = datetime.now()
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls_remaining,
            n_initial_points=n_initial_points_new,
            acq_func='EI',
            random_state=self.random_state,
            x0=x0,
            y0=y0,
            verbose=True
        )
        
        duration = datetime.now() - start_time
        logger.info(f"\nOptimization completed in: {duration}")

        # Cleanup checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"Checkpoint removed: {checkpoint_file}")

        # Extract best parameters
        best_params_list = result.x
        param_names = [dim.name for dim in space]
        best_params_dict = dict(zip(param_names, best_params_list))

        self.best_params = {k: type(v)(best_params_dict[k]) 
                           for k, v in self.best_params.items()}

        logger.info(f"\n{'='*80}")
        logger.info(f"Best Loss: {result.fun:.6f}")
        logger.info(f"Best Parameters: {self.best_params}")
        logger.info(f"{'='*80}")

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

    def save_results(self, results, filename=None):
        """Save results with convergence plots to bilstm_hpo_results directory"""
        # Create output directory
        output_dir = 'bilstm_hpo_results_512'
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = os.path.join(output_dir, f"c3_bilstm_fe_hpo_results_{timestamp}.json")
            filename = os.path.join(output_dir, f"{OUTPUT_PREFIX}_{timestamp}.json")
        else:
            filename = os.path.join(output_dir, filename)

        # Prepare JSON-serializable results
        json_results = []
        # for r in results['all_results;']:
        for r in results['all_results']:  # ✅ Correct
            json_results.append({
                'params': r['params'],
                'avg_val_loss': float(r['avg_val_loss']) if r['avg_val_loss'] < 1e10 else None,
                'fold_losses': [float(x) for x in r.get('fold_losses', [])],
                'avg_train_time': r.get('avg_train_time', 0),
                'n_folds_completed': r.get('n_folds_completed', 0)
            })

        final_results = {
            'optimization_method': 'Bayesian_GP',
            'model_type': 'Single_Layer_BiLSTM_with_13_features',
            'architecture': {
                'num_layers': 1,
                'lstm_activation': 'tanh',
                'epochs': 100,
                'merge_mode': 'concat'
            },
            'best_params': results['best_params'],
            'best_loss': float(results['best_loss']),
            'search_timestamp': datetime.now().isoformat(),
            'total_evaluations': len(json_results),
            'scaler_type': self.scaler_type,
            'sequence_length': self.sequence_length,
            'n_folds': len(self.data_splits),
            'feature_engineering': {
                'n_features': 13,
                'feature_types': ['raw', 'temporal_cyclical', 'derivative', 'volatility', 'statistical']
            },
            'convergence_data': {
                'func_vals': [float(v) for v in results['optimization_result'].func_vals],
                'evaluation_order': list(range(len(results['optimization_result'].func_vals)))
            },
            'all_results': json_results,
            'environment': {
                'tensorflow_version': tf.__version__,
                'numpy_version': np.__version__,
                'python_version': sys.version,
                'random_seed': RANDOM_SEED
            }
        }

        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {filename}")
        
        # Generate convergence plot (save in same directory)
        plot_file = filename.replace('.json', '_convergence.png')
        self.plot_convergence(results, plot_file)
        
        return filename

    def plot_convergence(self, results, filename):
        """Plot optimization convergence"""
        func_vals = results['convergence_data']['func_vals']
        valid_vals = [v for v in func_vals if v < 1e10]
        
        if not valid_vals:
            logger.warning("No valid values to plot")
            return
        
        cumulative_min = np.minimum.accumulate(valid_vals)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss per evaluation
        ax1.plot(valid_vals, 'o-', alpha=0.6, label='Loss per evaluation')
        ax1.plot(cumulative_min, 'r-', linewidth=2, label='Best loss so far')
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss distribution
        ax2.hist(valid_vals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(results['best_loss'], color='red', linestyle='--', 
                   linewidth=2, label=f'Best: {results["best_loss"]:.6f}')
        ax2.set_xlabel('Validation Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Loss Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Convergence plot saved to {filename}")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_hyperparameter_optimization_from_files(sequence_length=144, n_calls=50,
                                              scaler_type='standard', n_folds=5,
                                              random_state=42):
    """Run hyperparameter optimization with all improvements applied"""
    logger.info("="*80)
    logger.info("BAYESIAN HYPERPARAMETER OPTIMIZATION - BiLSTM with Feature Engineering")
    logger.info("="*80)
    
    try:
        csv_path = CSV_PATH
        if not os.path.exists(csv_path):
            logger.error(f"ERROR: '{csv_path}' not found.")
            return None
        
        # Load and validate data
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check for missing values
        if df['temp_value'].isna().any():
            logger.warning("Missing values detected in temp_value - filling with forward fill")
            df['temp_value'].fillna(method='ffill', inplace=True)
        
        logger.info(f"Loaded data: {len(df)} samples")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Value range: [{df['temp_value'].min():.2f}, {df['temp_value'].max():.2f}]")

        # Create expanding window splits (using raw data - feature engineering happens per-fold)
        data_splits = create_expanding_window_splits(
            df, sequence_length=sequence_length, n_folds=n_folds, purge_mult=2
        )
        
        if not data_splits:
            logger.error("ERROR: No valid cross-validation splits created")
            return None

        # Run optimization
        optimizer = HyperparameterOptimizer(
            data_splits, 
            sequence_length, 
            scaler_type,
            n_calls=n_calls,
            random_state=random_state
        )
        
        start_time = datetime.now()
        results = optimizer.bayesian_optimization()
        duration = datetime.now() - start_time
        
        logger.info(f"\nTotal optimization time: {duration}")

        # Save results
        results_file = optimizer.save_results(results)

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Model: Single-layer BiLSTM with 13-feature engineering")
        logger.info(f"Architecture: num_layers=1, activation='tanh', epochs=100")
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Scaler type: {scaler_type}")
        logger.info(f"Number of folds: {len(data_splits)}")
        logger.info(f"Total evaluations: {len(results['all_results'])}")
        logger.info(f"Best validation loss: {results['best_loss']:.6f}")
        logger.info(f"Best hyperparameters:")
        for k, v in results['best_params'].items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Results file: {results_file}")
        logger.info(f"{'='*80}")

        return {
            'results': results,
            'scaler_type': scaler_type,
            'sequence_length': sequence_length,
            'files': {'results': results_file}
        }
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("BiLSTM BAYESIAN OPTIMIZATION WITH FEATURE ENGINEERING")
    logger.info("="*80)
    
    try:
        csv_path = CSV_PATH
        if not os.path.exists(csv_path):
            logger.error(f"ERROR: '{csv_path}' not found.")
            logger.info("Please ensure the training data file exists at the specified path.")
        else:
            # Run hyperparameter optimization
            logger.info("\n" + "="*80)
            logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
            logger.info("="*80)
            
            results_dict = run_hyperparameter_optimization_from_files(
                sequence_length=24,
                n_calls=25,
                scaler_type='standard',
                n_folds=5,
                random_state=42
            )
            
            if results_dict is not None:
                logger.info("\n" + "="*80)
                logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
                logger.info("="*80)
                logger.info(f"Best validation loss: {results_dict['results']['best_loss']:.6f}")
                logger.info(f"Best parameters: {results_dict['results']['best_params']}")

            else:
                logger.error("Optimization failed. Check logs for details.")
                
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user.")
        logger.info("Progress has been checkpointed and can be resumed.")
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()