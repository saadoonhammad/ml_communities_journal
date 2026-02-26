# ============================================================================
# MLP AUTOENCODER - FINAL TRAINING SCRIPT
# ============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
import json
import logging
import warnings
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print("="*80)
print("MLP AUTOENCODER - FINAL TRAINING")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# GPU Configuration
TARGET_GPU = 0  # Change to 0 for other GPU
gpus = tf.config.list_physical_devices('GPU')
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
    """Add cyclical temporal features to preserve periodicity."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Cyclical encoding for hour (24-hour period)
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    
    # Day of Week (Period=7)
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
    
    # Fill NaN values
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
    df['static_sensor_alert'] = np.log1p(df['static_sensor_alert'])

    return df

def add_statistical_features(sequences):
    """
    Add statistical features computed over each sequence window.
    Input: (n_samples, sequence_length, n_base_features)
    Output: (n_samples, sequence_length, n_base_features + 3)
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

def prepare_data_with_features(df, sequence_length, feature_mode='all'):
    """
    Prepare data with different feature configurations
    
    feature_mode options:
    - 'raw': Only temperature sequences
    - 'temporal': Temperature + temporal context (hour, day)
    - 'statistical': Temperature + statistical features (mean, std, range)
    - 'all': Temperature + temporal + derivative + volatility + statistical
    
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
        feature_names: list of feature names
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PREPARING DATA - Mode: {feature_mode}")
    logger.info(f"{'='*60}")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle missing values
    if df['temp_value'].isna().any():
        logger.warning("Missing values found. Filling with forward fill.")
        df['temp_value'].fillna(method='ffill', inplace=True)
    
    if feature_mode == 'raw':
        # Just temperature sequences
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        feature_names = ['temp_value']
        
    elif feature_mode == 'temporal':
        # Temperature + temporal features
        df = add_temporal_features(df)
        
        feature_cols = ['temp_value', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        feature_names = feature_cols
        
    elif feature_mode == 'statistical':
        # Temperature + statistical features
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = ['temp_value', 'mean', 'std', 'range']

    elif feature_mode == 'derivative':
        # Temperature + derivative features
        df = add_derivative_features(df)
        feature_cols = ['temp_value', 'energy', 'velocity', 'acceleration', 'roll_std']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = feature_cols + ['mean', 'std', 'range']

    elif feature_mode == 'volatility':
        # Temperature + volatility features
        df = add_volatility_features(df)
        feature_cols = ['temp_value', 'static_sensor_alert']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        feature_names = feature_cols     

    elif feature_mode == 'all':
        # All features (13 features total)
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
    plt.title('MLP Autoencoder Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training plot saved: {save_path}")

# ============================================================================
# MLP AUTOENCODER MODEL CLASS
# ============================================================================

class MLPAutoencoder:
    """Multi-Layer Perceptron Autoencoder for anomaly detection"""
    
    def __init__(self, sequence_length, n_features, 
                 hidden_units=128, encoding_dim=32, 
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
        """Build MLP Autoencoder architecture"""
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
        
        logger.info("\n" + "="*60)
        logger.info("MLP AUTOENCODER ARCHITECTURE")
        logger.info("="*60)
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, patience=15, patience_factor=0.2):
        """Train the autoencoder with patience factor"""
        if self.model is None:
            self.build_model()
        
        # Calculate dynamic patience based on patience factor
        dynamic_patience = max(5, int(patience_factor * epochs))
        lr_patience = max(3, int(dynamic_patience / 3))
        
        logger.info(f"\nTraining parameters:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Early stopping patience: {dynamic_patience} (factor: {patience_factor})")
        logger.info(f"  LR reduction patience: {lr_patience}")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=dynamic_patience, 
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=lr_patience, 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING MLP AUTOENCODER")
        logger.info("="*60)
        
        self.history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Preserve chronological order
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

CLUSTER = 'c3'
TRAIN_CSV_PATH = f'../1. Datasets/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_DIR = f'../Final Train/Representative/mlp_trained_model/{CLUSTER}'

# Data parameters
SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8  # 80% train, 20% validation

# Model parameters (from BOHB optimization)
HIDDEN_UNITS = 162    
ENCODING_DIM = 70  
DROPOUT_RATE = 0.10087485185775767  
LEARNING_RATE = 0.00014177530666893253 
BATCH_SIZE = 32     
EPOCHS = 150        
PATIENCE_FACTOR = 0.19184106107344448 

# Feature mode: 'raw', 'temporal', 'statistical', 'derivative', 'volatility', 'all'
FEATURE_MODE = 'all'

# Threshold method
N_SIGMA = 3  # μ + 3σ threshold

# ============================================================================
# LOAD AND PREPARE TRAINING DATA
# ============================================================================

logger.info("\n" + "="*60)
logger.info("LOADING TRAINING DATA")
logger.info("="*60)

if not os.path.exists(TRAIN_CSV_PATH):
    logger.error(f"ERROR: '{TRAIN_CSV_PATH}' not found!")
    exit(1)

df_train = pd.read_csv(TRAIN_CSV_PATH)
logger.info(f"Loaded {len(df_train)} samples")
logger.info(f"Columns: {df_train.columns.tolist()}")
logger.info(f"\nDate range: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")

# Prepare sequences with features
sequences, feature_names = prepare_data_with_features(
    df_train, SEQUENCE_LENGTH, feature_mode=FEATURE_MODE
)

n_features = sequences.shape[2]
logger.info(f"\nTotal features: {n_features}")

# ========================================================================
# TRAIN/VAL SPLIT (80/20)
# ========================================================================

split_idx = int(len(sequences) * TRAIN_RATIO)
X_train_raw = sequences[:split_idx]
X_val_raw = sequences[split_idx:]

logger.info(f"\n{'='*60}")
logger.info("DATA SPLIT (80/20)")
logger.info(f"{'='*60}")
logger.info(f"Total sequences: {len(sequences)}")
logger.info(f"Train: {len(X_train_raw)} ({len(X_train_raw)/len(sequences)*100:.1f}%)")
logger.info(f"Val:   {len(X_val_raw)} ({len(X_val_raw)/len(sequences)*100:.1f}%)")

# ========================================================================
# SCALING (fit on train only)
# ========================================================================

logger.info("\n" + "="*60)
logger.info("SCALING DATA")
logger.info("="*60)

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
    logger.info(f"Feature {i} ({feature_names[i]}): mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")

# ========================================================================
# BUILD AND TRAIN MODEL
# ========================================================================

autoencoder = MLPAutoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=n_features,
    hidden_units=HIDDEN_UNITS,
    encoding_dim=ENCODING_DIM,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE
)

history = autoencoder.train(
    X_train_scaled, X_val_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience_factor=PATIENCE_FACTOR
)

# ========================================================================
# CALCULATE RECONSTRUCTION ERRORS ON TRAINING DATA
# ========================================================================

logger.info("\n" + "="*60)
logger.info("CALCULATING RECONSTRUCTION ERRORS")
logger.info("="*60)

errors_train = autoencoder.get_reconstruction_errors(X_train_scaled)
errors_val = autoencoder.get_reconstruction_errors(X_val_scaled)

logger.info(f"\nTrain errors: mean={errors_train.mean():.6f}, std={errors_train.std():.6f}")
logger.info(f"Val errors:   mean={errors_val.mean():.6f}, std={errors_val.std():.6f}")

# ========================================================================
# DETERMINE THRESHOLD (μ + 3σ on training data)
# ========================================================================

logger.info("\n" + "="*60)
logger.info("DETERMINING THRESHOLD (μ + 3σ)")
logger.info("="*60)

# Calculate mean and standard deviation of training errors
mu_train = errors_train.mean()
sigma_train = errors_train.std()

# Threshold = μ + 3σ (captures ~99.7% of normal data if normally distributed)
threshold = mu_train + N_SIGMA * sigma_train

logger.info(f"\nTraining Error Statistics:")
logger.info(f"  Mean (μ):        {mu_train:.6f}")
logger.info(f"  Std Dev (σ):     {sigma_train:.6f}")
logger.info(f"  Min:             {errors_train.min():.6f}")
logger.info(f"  Max:             {errors_train.max():.6f}")
logger.info(f"  Median:          {np.median(errors_train):.6f}")
logger.info(f"\nThreshold (μ + {N_SIGMA}σ): {threshold:.6f}")

# Expected false positive rate (assuming normal distribution)
expected_fpr = (errors_train > threshold).sum() / len(errors_train)
logger.info(f"Expected FPR on train: {expected_fpr:.4f} ({expected_fpr*100:.2f}%)")

# Check anomaly rate in validation
anomalies_val = errors_val > threshold
n_anomalies_val = anomalies_val.sum()
anomaly_rate_val = (n_anomalies_val / len(errors_val)) * 100
logger.info(f"\nValidation anomaly rate: {n_anomalies_val}/{len(errors_val)} ({anomaly_rate_val:.2f}%)")

# ========================================================================
# VISUALIZE TRAINING
# ========================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot training history
plot_training_history(history, f'{OUTPUT_DIR}/training_history_{CLUSTER}.png')

# Plot reconstruction errors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(errors_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
ax1.hist(errors_val, bins=50, alpha=0.7, label='Validation', edgecolor='black')
ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Threshold: {threshold:.4f}')
ax1.set_xlabel('Reconstruction Error (MSE)')
ax1.set_ylabel('Frequency')
ax1.set_title('Reconstruction Error Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter
ax2.scatter(range(len(errors_train)), errors_train, alpha=0.5, s=10, label='Train')
ax2.scatter(range(len(errors_val)), errors_val, alpha=0.5, s=10, label='Validation')
ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Threshold: {threshold:.4f}')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Reconstruction Error (MSE)')
ax2.set_title('Reconstruction Errors')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/reconstruction_errors_train_val_{CLUSTER}.png', dpi=150, bbox_inches='tight')
plt.close()

# ========================================================================
# SAVE MODEL, SCALERS, AND METADATA
# ========================================================================

logger.info("\n" + "="*60)
logger.info("SAVING MODEL AND ARTIFACTS")
logger.info("="*60)

# Save model
autoencoder.model.save(f'{OUTPUT_DIR}/mlp_autoencoder_{CLUSTER}.h5')
logger.info(f"✓ Model saved: {OUTPUT_DIR}/mlp_autoencoder_{CLUSTER}.h5")

# Save scalers
joblib.dump(scalers, f'{OUTPUT_DIR}/scalers_{CLUSTER}.pkl')
logger.info(f"✓ Scalers saved: {OUTPUT_DIR}/scalers_{CLUSTER}.pkl")

# Save metadata
metadata = {
    'model_type': 'MLP_Autoencoder',
    'cluster': CLUSTER,
    'feature_mode': FEATURE_MODE,
    'feature_names': feature_names,
    'n_features': n_features,
    'sequence_length': SEQUENCE_LENGTH,
    'model_parameters': {
        'hidden_units': HIDDEN_UNITS,
        'encoding_dim': ENCODING_DIM,
        'dropout_rate': DROPOUT_RATE,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE
    },
    'training_parameters': {
        'epochs': EPOCHS,
        'patience_factor': PATIENCE_FACTOR,
        'train_ratio': TRAIN_RATIO,
        'random_seed': RANDOM_SEED
    },
    'threshold': {
        'value': float(threshold),
        'method': f'mu_plus_{N_SIGMA}_sigma',
        'mu': float(mu_train),
        'sigma': float(sigma_train),
        'n_sigma': N_SIGMA
    },
    'data_split': {
        'train_samples': len(X_train_scaled),
        'val_samples': len(X_val_scaled),
        'train_ratio': TRAIN_RATIO
    },
    'training_results': {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_loss']))
    },
    'error_statistics': {
        'train_mean': float(errors_train.mean()),
        'train_std': float(errors_train.std()),
        'train_min': float(errors_train.min()),
        'train_max': float(errors_train.max()),
        'val_mean': float(errors_val.mean()),
        'val_std': float(errors_val.std()),
        'val_min': float(errors_val.min()),
        'val_max': float(errors_val.max())
    },
    'anomaly_rates': {
        'expected_fpr_train': float(expected_fpr),
        'anomaly_rate_val': float(anomaly_rate_val/100),
        'n_anomalies_val': int(n_anomalies_val),
        'total_val': len(errors_val)
    },
    'timestamp': datetime.now().isoformat(),
    'environment': {
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__
    }
}

with open(f'{OUTPUT_DIR}/metadata_{CLUSTER}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
logger.info(f"✓ Metadata saved: {OUTPUT_DIR}/metadata_{CLUSTER}.json")

# Save training history as CSV
history_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'train_mae': history.history.get('mae', [np.nan] * len(history.history['loss'])),
    'val_mae': history.history.get('val_mae', [np.nan] * len(history.history['loss']))
})
history_df.to_csv(f'{OUTPUT_DIR}/training_history_{CLUSTER}.csv', index=False)
logger.info(f"✓ Training history saved: {OUTPUT_DIR}/training_history_{CLUSTER}.csv")

logger.info(f"✓ Reconstruction errors saved: {OUTPUT_DIR}/reconstruction_errors_{CLUSTER}.csv")

logger.info("\n" + "="*60)
logger.info("TRAINING COMPLETE!")
logger.info("="*60)
logger.info(f"\nModel artifacts saved in: {OUTPUT_DIR}/")
logger.info("Files created:")
logger.info("  - mlp_autoencoder.h5 (trained model)")
logger.info("  - scalers.pkl (feature scalers)")
logger.info("  - metadata.json (configuration and results)")
logger.info("  - training_history.png")
logger.info("  - training_history.csv")
logger.info("  - reconstruction_errors_train_val.png")
logger.info("  - reconstruction_errors.csv")