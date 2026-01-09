# ============================================================================
# MODEL TRAINING
# ============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
import json
import logging
import warnings


logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

TARGET_GPU = 0  

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sequences(data, sequence_length):
    """Create sliding window sequences"""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def add_temporal_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Cyclical Encoding (The Fix) ---
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    
    # Optional: Day of Week (Period=7)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    
    # Optional: Month (Period=12)
    # df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12.0)
    # df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12.0)
  
    return df

def add_derivative_features(df, column='temp_value', window=1):
    """Add derivative-based features"""
    df = df.copy()
    
    df['velocity'] = df[column].diff(window)
    df['acceleration'] = df['velocity'].diff(window)
    df['energy'] = df[column]**2
    df['roll_std'] = df[column].rolling(window=144).std()
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

def add_volatility_features(df, column='temp_value', window=10):
    df = df.copy()
    rolling_std = df[column].rolling(window=10).std()
    rolling_std.fillna(method='bfill', inplace=True)
    epsilon = 1e-3 
    df['static_sensor_alert'] = 1.0 / (rolling_std + epsilon)

    # Optional: Log transform to squash the infinity slightly if it breaks training, 
    # but usually, the raw inverse is better for anomaly detection.
    df['static_sensor_alert'] = np.log1p(df['static_sensor_alert'])

    return df

def add_statistical_features(sequences):
    """
    Add statistical features to each sequence
    Input: (n_samples, sequence_length, n_base_features)
    Output: (n_samples, sequence_length, n_base_features + 3)
    
    For each sequence, calculates: mean, std, range
    These values are repeated across all timesteps in the sequence
    """
    n_samples = sequences.shape[0]
    seq_len = sequences.shape[1]
    n_base_features = sequences.shape[2]
    
    stats_features = []
    
    for seq in sequences:
        # Calculate stats only on the first feature (temp_value)
        temp_values = seq[:, 0]  # Take only temperature column
        
        mean_val = np.mean(temp_values)
        std_val = np.std(temp_values)
        range_val = np.max(temp_values) - np.min(temp_values)
        
        # Create feature array (repeat stats for each timestep)
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

def prepare_data_with_features(df, sequence_length, feature_mode='raw'):
    """
    Prepare data with different feature configurations
    
    feature_mode options:
    - 'raw': Only temperature sequences
    - 'temporal': Temperature + temporal context (hour, day, month)
    - 'statistical': Temperature + statistical features (mean, std, range)
    - 'all': Temperature + temporal + statistical
    
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
        feature_names: list of feature names
    """
    print(f"\n{'='*60}")
    print(f"PREPARING DATA - Mode: {feature_mode}")
    print(f"{'='*60}")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle missing values
    if df['temp_value'].isna().any():
        print("Warning: Missing values found. Filling with forward fill.")
        df['temp_value'].fillna(method='ffill', inplace=True)
    
    if feature_mode == 'raw':
        # Just temperature sequences
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        feature_names = ['temp_value']
        
    elif feature_mode == 'temporal':
        # Temperature + temporal features
        df = add_temporal_features(df)
        
        feature_cols = ['temp_value', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] #'month_sin', 'month_cos'
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        feature_names = feature_cols
        
    elif feature_mode == 'statistical':
        # Temperature + statistical features
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = ['temp_value', 'mean', 'std', 'range']

    elif feature_mode== 'derivative':
        df = add_derivative_features(df)
        feature_cols = ['temp_value', 'energy', 'velocity', 'acceleration']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = feature_cols + ['mean', 'std', 'range']

    elif feature_mode=='volatility':

        df= add_volatility_features(df)
        feature_cols = ['temp_value', 'static_sensor_alert']

        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        feature_names = feature_cols     

    elif feature_mode == 'all':
        df = add_temporal_features(df)
        df = add_derivative_features(df)
        df=add_volatility_features(df)
        
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
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Features ({sequences.shape[2]}): {feature_names}")
    
    return sequences, feature_names

def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    plt.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
                label=f'Best Epoch: {best_epoch+1}')
    plt.plot(best_epoch, best_val_loss, 'r*', markersize=15)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training plot saved: {save_path}")


# ============================================================================
# MODEL CLASS
# ============================================================================

class SimpleBiLSTMAutoencoder:
    """Simple Bi-LSTM Autoencoder for anomaly detection"""
    
    def __init__(self, sequence_length, n_features, 
                 lstm_units=64, encoding_dim=32, learning_rate=0.001, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build simple 1-layer Bi-LSTM Autoencoder"""
        inp = Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        # Encoder
        encoded = Bidirectional(
            LSTM(self.lstm_units, activation='tanh', dropout=self.dropout_rate, return_sequences=False),
            merge_mode='concat'
        )(inp)
        
        # # Bottleneck
        z = Dense(self.encoding_dim, activation='tanh')(encoded)
        
        # # Decoder
        decoded = RepeatVector(self.sequence_length)(z)
        decoded = Bidirectional(
            LSTM(self.lstm_units, activation='tanh', dropout=self.dropout_rate, return_sequences=True),
            merge_mode='concat'
        )(decoded)
        
        # Output
        out = TimeDistributed(Dense(self.n_features))(decoded)
        
        self.model = Model(inp, out)
        optimizer = Adamax(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, patience=15):
        """Train the autoencoder"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=5, min_lr=1e-7, verbose=1)
        ]
        
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        self.history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        return self.history
    
    def predict(self, X):
        """Reconstruct input data"""
        return self.model.predict(X, verbose=0)
    
    def get_reconstruction_errors(self, X):
        """Calculate reconstruction errors (MSE per sample)"""
        preds = self.predict(X)
        return np.mean(np.square(X - preds), axis=(1, 2))


# ============================================================================
# CONFIGURATION
# ============================================================================

CLUSTER= 'c4'
TRAIN_CSV_PATH = f'1. Datasets/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_DIR = f'lstm_trained_model/{CLUSTER}'

# Data parameters
SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8  # 80% train, 20% validation

# Model parameters
LSTM_UNITS = 256
ENCODING_DIM = 64
DROPOUT_RATE=0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 200 
PATIENCE = 15

# Feature mode: 'raw', 'temporal', 'statistical', 'all'
FEATURE_MODE = 'all'

# Threshold for anomaly detection
# THRESHOLD_PERCENTILE = 99
N_SIGMA=3


# ============================================================================
# LOAD AND PREPARE TRAINING DATA
# ============================================================================

print("\n" + "="*60)
print("LOADING TRAINING DATA")
print("="*60)

if not os.path.exists(TRAIN_CSV_PATH):
    print(f"ERROR: '{TRAIN_CSV_PATH}' not found!")
else:
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(df_train)} samples")
    print(f"Columns: {df_train.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_train.head())
    
    # Prepare sequences with features
    sequences, feature_names = prepare_data_with_features(
        df_train, SEQUENCE_LENGTH, feature_mode=FEATURE_MODE
    )
    
    n_features = sequences.shape[2]
    print(f"\nTotal features: {n_features}")
    # print(sequences)
    
    
    # ========================================================================
    # TRAIN/VAL SPLIT (80/20)
    # ========================================================================
    
    split_idx = int(len(sequences) * TRAIN_RATIO)
    X_train_raw = sequences[:split_idx]
    X_val_raw = sequences[split_idx:]
    
    print(f"\n{'='*60}")
    print("DATA SPLIT (80/20)")
    print(f"{'='*60}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Train: {len(X_train_raw)} ({len(X_train_raw)/len(sequences)*100:.1f}%)")
    print(f"Val:   {len(X_val_raw)} ({len(X_val_raw)/len(sequences)*100:.1f}%)")
    
    
    # ========================================================================
    # SCALING (fit on train only)
    # ========================================================================
    
    print("\n" + "="*60)
    print("SCALING DATA")
    print("="*60)
    
    scalers = []
    X_train_scaled = X_train_raw.copy()
    X_val_scaled = X_val_raw.copy()
    
    for i in range(n_features):
        scaler = StandardScaler()
        
        # Fit ONLY on training data
        X_train_scaled[:, :, i] = scaler.fit_transform(
            X_train_raw[:, :, i].reshape(-1, 1)
        ).reshape(X_train_raw.shape[0], SEQUENCE_LENGTH)
        
        # Transform validation data
        X_val_scaled[:, :, i] = scaler.transform(
            X_val_raw[:, :, i].reshape(-1, 1)
        ).reshape(X_val_raw.shape[0], SEQUENCE_LENGTH)
        
        scalers.append(scaler)
        print(f"Feature {i} ({feature_names[i]}): mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    
    # ========================================================================
    # BUILD AND TRAIN MODEL
    # ========================================================================
    
    autoencoder = SimpleBiLSTMAutoencoder(
        sequence_length=SEQUENCE_LENGTH,
        n_features=n_features,
        lstm_units=LSTM_UNITS,
        encoding_dim=ENCODING_DIM,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE
    )
    
    history = autoencoder.train(
        X_train_scaled, X_val_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )
    
    # ========================================================================
    # CALCULATE RECONSTRUCTION ERRORS ON TRAINING DATA
    # ========================================================================
    
    print("\n" + "="*60)
    print("CALCULATING RECONSTRUCTION ERRORS")
    print("="*60)
    
    errors_train = autoencoder.get_reconstruction_errors(X_train_scaled)
    errors_val = autoencoder.get_reconstruction_errors(X_val_scaled)
    
    print(f"\nTrain errors: mean={errors_train.mean():.6f}, std={errors_train.std():.6f}")
    print(f"Val errors:   mean={errors_val.mean():.6f}, std={errors_val.std():.6f}")


    # ========================================================================
    # DETERMINE THRESHOLD (μ + 3σ on training data)
    # ========================================================================

    print("\n" + "="*60)
    print("DETERMINING THRESHOLD (μ + 3σ)")
    print("="*60)
    
    # Calculate mean and standard deviation of training errors
    mu_train = errors_train.mean()
    sigma_train = errors_train.std()
    
    # Threshold = μ + 3σ (captures ~99.7% of normal data if normally distributed)
    threshold = mu_train + 3 * sigma_train
    
    print(f"\nTraining Error Statistics:")
    print(f"  Mean (μ):        {mu_train:.6f}")
    print(f"  Std Dev (σ):     {sigma_train:.6f}")
    print(f"  Min:             {errors_train.min():.6f}")
    print(f"  Max:             {errors_train.max():.6f}")
    print(f"  Median:          {np.median(errors_train):.6f}")
    print(f"\nThreshold (μ + 3σ): {threshold:.6f}")
    
    # Expected false positive rate (assuming normal distribution)
    expected_fpr = (errors_train > threshold).sum() / len(errors_train)
    print(f"Expected FPR on train: {expected_fpr:.4f} ({expected_fpr*100:.2f}%)")
    
    # Check anomaly rate in validation
    anomalies_val = errors_val > threshold
    n_anomalies_val = anomalies_val.sum()
    anomaly_rate_val = (n_anomalies_val / len(errors_val)) * 100
    print(f"\nValidation anomaly rate: {n_anomalies_val}/{len(errors_val)} ({anomaly_rate_val:.2f}%)")
    
    
    # ========================================================================
    # VISUALIZE TRAINING
    # ========================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training_history(history, f'{OUTPUT_DIR}/training_history_{CLUSTER}.png')
    
    # Plot reconstruction errors
    plt.figure(figsize=(14, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
    plt.hist(errors_val, bins=50, alpha=0.7, label='Validation', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(errors_train)), errors_train, alpha=0.5, s=10, label='Train')
    plt.scatter(range(len(errors_val)), errors_val, alpha=0.5, s=10, label='Validation')
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/reconstruction_errors_train_val_{CLUSTER}.png', dpi=150, bbox_inches='tight')
    plt.show()    
    
    # ========================================================================
    # SAVE MODEL, SCALERS, AND METADATA
    # ========================================================================
    
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    # Save model
    autoencoder.model.save(f'{OUTPUT_DIR}/lstm_autoencoder_{CLUSTER}.h5')
    print(f"✓ Model saved: {OUTPUT_DIR}/lstm_autoencoder_{CLUSTER}.h5")
    
    # Save scalers
    joblib.dump(scalers, f'{OUTPUT_DIR}/scalers_{CLUSTER}.pkl')
    print(f"✓ Scalers saved: {OUTPUT_DIR}/scalers_{CLUSTER}.pkl")
    
    # Save metadata
    metadata = {
        'feature_mode': FEATURE_MODE,
        'feature_names': feature_names,
        'n_features': n_features,
        'sequence_length': SEQUENCE_LENGTH,
        'lstm_units': LSTM_UNITS,
        'encoding_dim': ENCODING_DIM,
        # 'threshold': float(threshold),
        # 'threshold_percentile': THRESHOLD_PERCENTILE,
        'threshold': float(threshold),
        'threshold_method': 'mu_plus_3sigma',
        'threshold_stats': {
            'mu': float(mu_train),
            'sigma': float(sigma_train),
            'n_sigma': 3
        },
        'data_split': {
            'train_samples': len(X_train_scaled),
            'val_samples': len(X_val_scaled),
            'train_ratio': TRAIN_RATIO
        },
        'training': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            'epochs_trained': len(history.history['loss'])
        },
        'errors': {
            'train_mean': float(errors_train.mean()),
            'train_std': float(errors_train.std()),
            'val_mean': float(errors_val.mean()),
            'val_std': float(errors_val.std())
        }
    }
    
    with open(f'{OUTPUT_DIR}/metadata_{CLUSTER}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {OUTPUT_DIR}/metadata_{CLUSTER}.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel artifacts saved in: {OUTPUT_DIR}/")
    print("Files created:")
    print("  - bilstm_autoencoder.h5 (trained model)")
    print("  - scalers.pkl (feature scalers)")
    print("  - metadata.json (configuration and results)")
    print("  - training_history.png")
    print("  - reconstruction_errors_train_val.png")