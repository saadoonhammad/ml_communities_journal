# ============================================================================
# PART 1: DATA PREPARATION AND MODEL TRAINING - MULTI-SENSOR BiLSTM AUTOENCODER
# ============================================================================
# INCLUDES PER-SENSOR THRESHOLDING FOR LOCALIZATION

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, LSTM, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.optimizers import Adam
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

# GPU Configuration
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
# CONFIGURATION
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
TRAIN_CSV_PATH = f'../Global_Network/Train Sets/{CLUSTER}_train/train_data_{CLUSTER}.csv'
OUTPUT_DIR = f'../Final Train/Network Global/bilstm_trained_model/'


# Data parameters
SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8  # 80% train, 20% validation

# Model parameters from BOHB optimization
HIDDEN_UNITS = 127
ENCODING_DIM = 94
DROPOUT_RATE = 0.10108005586297307
LEARNING_RATE = 0.00015203990358413587
BATCH_SIZE = 32
EPOCHS = 74
PATIENCE = 15  

# Additional BOHB parameter
PATIENCE_FACTOR = 0.22645550935586198
# Threshold for anomaly detection
N_SIGMA = 3


# ============================================================================
# HELPER FUNCTIONS FOR MULTI-SENSOR DATA
# ============================================================================

def create_sequences(data, sequence_length):
    """Create sliding window sequences preserving chronological order."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def prepare_multisensor_data(df, sequence_length, sensor_columns):
    """
    Prepare multi-sensor data for BiLSTM autoencoder.
    Returns sequences and sensor names
    """
    print(f"\n{'='*60}")
    print("PREPARING MULTI-SENSOR DATA")
    print(f"{'='*60}")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for missing values
    missing_sensors = [sensor for sensor in sensor_columns if sensor not in df.columns]
    if missing_sensors:
        raise ValueError(f"Missing sensor columns in data: {missing_sensors}")
    
    # Handle missing values
    for sensor in sensor_columns:
        if df[sensor].isna().any():
            print(f"Warning: Missing values in {sensor}. Filling with forward fill.")
            df[sensor].fillna(method='ffill', inplace=True)
    
    # Use only sensor values
    sensor_data = df[sensor_columns].values
    sequences = create_sequences(sensor_data, sequence_length)
    feature_names = sensor_columns
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Sensors ({sequences.shape[2]}): {feature_names}")
    
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
# BiLSTM AUTOENCODER MODEL CLASS
# ============================================================================

class FusedBiLSTMAutoencoder:
    """Bidirectional LSTM Autoencoder for multi-sensor time series"""
    
    def __init__(self, sequence_length, n_sensors,
                 hidden_units=64, encoding_dim=32, 
                 learning_rate=0.001, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_sensors = n_sensors
        self.hidden_units = hidden_units
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the BiLSTM Autoencoder architecture"""
        inp = Input(shape=(self.sequence_length, self.n_sensors))
        
        # Encoder: Bidirectional LSTM returns concatenated last states
        # Output shape: (batch_size, 2 * hidden_units)
        x = Bidirectional(
            LSTM(units=self.hidden_units, 
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate,
                 return_sequences=False)
        )(inp)
        
        # Bottleneck: Dense layer with tanh activation (like original MLP bottleneck)
        z = Dense(self.encoding_dim, activation='tanh')(x)
        
        # Decoder: Repeat vector to restore sequence dimension
        x = RepeatVector(self.sequence_length)(z)
        
        # Decoder: Bidirectional LSTM returns sequences
        # Output shape: (batch_size, sequence_length, 2 * hidden_units)
        x = Bidirectional(
            LSTM(units=self.hidden_units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate,
                 return_sequences=True)
        )(x)
        
        # Output layer: reconstruct each timestep (reduce to original n_sensors)
        out = TimeDistributed(Dense(self.n_sensors))(x)
        
        self.model = Model(inp, out)
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("\n" + "="*60)
        print("BiLSTM AUTOENCODER ARCHITECTURE")
        print("="*60)
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, X_val, epochs=100, batch_size=32, 
              patience=15, callbacks=None):
        """Train the autoencoder on multi-sensor data"""
        if self.model is None:
            self.build_model()
        
        default_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=max(5, patience//3), 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        print("\n" + "="*60)
        print("TRAINING BiLSTM AUTOENCODER")
        print("="*60)
        
        self.history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        return self.history
    
    def predict(self, X):
        """Reconstruct input data"""
        return self.model.predict(X, verbose=0)
    
    def get_reconstruction_errors(self, X):
        """Calculate reconstruction errors (MSE per sample)"""
        preds = self.predict(X)
        return np.mean(np.square(X - preds), axis=(1, 2))
    
    def get_per_sensor_errors(self, X):
        """Calculate per-sensor reconstruction errors"""
        preds = self.predict(X)
        # Mean squared error per sensor (axis=1 is the sequence dimension)
        return np.mean(np.square(X - preds), axis=1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("LOADING TRAINING DATA FOR CLUSTER:", CLUSTER)
print("="*60)

if not os.path.exists(TRAIN_CSV_PATH):
    print(f"ERROR: '{TRAIN_CSV_PATH}' not found!")
else:
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(df_train)} samples")
    print(f"Columns: {df_train.columns.tolist()}")
    
    # Check if sensor columns exist
    available_sensors = [col for col in SENSOR_COLS if col in df_train.columns]
    if len(available_sensors) != len(SENSOR_COLS):
        missing = set(SENSOR_COLS) - set(available_sensors)
        print(f"Warning: Missing sensor columns: {missing}")
        print(f"Using available sensors: {available_sensors}")
        sensor_cols_to_use = available_sensors
    else:
        sensor_cols_to_use = SENSOR_COLS
    
    print(f"\nFirst few rows:")
    print(df_train[['timestamp'] + sensor_cols_to_use].head())
    
    # Prepare sequences for multi-sensor data
    sequences, sensor_names = prepare_multisensor_data(
        df_train, SEQUENCE_LENGTH, sensor_cols_to_use
    )
    
    n_features = sequences.shape[2]
    print(f"\nTotal sensors: {n_features}")
    print(f"Sensors: {sensor_names}")
    
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
    # SCALING (fit on train only, per sensor)
    # ========================================================================
    
    print("\n" + "="*60)
    print("SCALING DATA - PER SENSOR")
    print("="*60)
    
    scalers = []
    X_train_scaled = X_train_raw.copy()
    X_val_scaled = X_val_raw.copy()
    
    for i in range(n_features):
        scaler = StandardScaler()
        
        # Fit ONLY on training data for each sensor
        X_train_scaled[:, :, i] = scaler.fit_transform(
            X_train_raw[:, :, i].reshape(-1, 1)
        ).reshape(X_train_raw.shape[0], SEQUENCE_LENGTH)
        
        # Transform validation data using training statistics
        X_val_scaled[:, :, i] = scaler.transform(
            X_val_raw[:, :, i].reshape(-1, 1)
        ).reshape(X_val_raw.shape[0], SEQUENCE_LENGTH)
        
        scalers.append(scaler)
        print(f"Sensor {i} ({sensor_names[i]}): mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    
    # ========================================================================
    # BUILD AND TRAIN MODEL
    # ========================================================================
    
    # Calculate patience based on patience_factor from BOHB
    calculated_patience = max(5, int(PATIENCE_FACTOR * EPOCHS))
    print(f"\nCalculated patience: {calculated_patience} (from patience_factor={PATIENCE_FACTOR})")
    
    autoencoder = FusedBiLSTMAutoencoder(
        sequence_length=SEQUENCE_LENGTH,
        n_sensors=n_features,
        hidden_units=HIDDEN_UNITS,
        encoding_dim=ENCODING_DIM,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE
    )
    
    history = autoencoder.train(
        X_train_scaled, X_val_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=calculated_patience
    )
    
    # ========================================================================
    # CALCULATE RECONSTRUCTION ERRORS
    # ========================================================================
    
    print("\n" + "="*60)
    print("CALCULATING RECONSTRUCTION ERRORS")
    print("="*60)
    
    errors_train = autoencoder.get_reconstruction_errors(X_train_scaled)
    errors_val = autoencoder.get_reconstruction_errors(X_val_scaled)
    
    print(f"\nTrain errors: mean={errors_train.mean():.6f}, std={errors_train.std():.6f}")
    print(f"Val errors:   mean={errors_val.mean():.6f}, std={errors_val.std():.6f}")
    
    # ========================================================================
    # CALCULATE PER-SENSOR RECONSTRUCTION ERRORS
    # ========================================================================
    
    print("\n" + "="*60)
    print("CALCULATING PER-SENSOR RECONSTRUCTION ERRORS")
    print("="*60)
    
    # Get per-sensor errors for validation set
    per_sensor_errors_val = autoencoder.get_per_sensor_errors(X_val_scaled)
    print(f"Per-sensor validation errors shape: {per_sensor_errors_val.shape}")
    
    # Calculate per-sensor statistics
    per_sensor_stats = {}
    per_sensor_thresholds = {}
    
    print("\nPer-sensor statistics (validation set):")
    for i, sensor_name in enumerate(sensor_names):
        sensor_errors = per_sensor_errors_val[:, i]
        mu = np.mean(sensor_errors)
        sigma = np.std(sensor_errors)
        threshold = mu + N_SIGMA * sigma
        
        per_sensor_stats[sensor_name] = {
            'mean': float(mu),
            'std': float(sigma),
            'min': float(np.min(sensor_errors)),
            'max': float(np.max(sensor_errors)),
            'median': float(np.median(sensor_errors))
        }
        
        per_sensor_thresholds[sensor_name] = float(threshold)
        
        print(f"{sensor_name}:")
        print(f"  Mean (μ):    {mu:.6f}")
        print(f"  Std Dev (σ): {sigma:.6f}")
        print(f"  Threshold (μ + {N_SIGMA}σ): {threshold:.6f}")
    
    # ========================================================================
    # DETERMINE GLOBAL THRESHOLD (μ + 3σ on training data)
    # ========================================================================
    
    print("\n" + "="*60)
    print("DETERMINING GLOBAL THRESHOLD (μ + 3σ)")
    print("="*60)
    
    # Calculate mean and standard deviation of training errors
    mu_train = errors_train.mean()
    sigma_train = errors_train.std()
    
    # Threshold = μ + 3σ (captures ~99.7% of normal data if normally distributed)
    global_threshold = mu_train + N_SIGMA * sigma_train
    
    print(f"\nGlobal Error Statistics (Training):")
    print(f"  Mean (μ):        {mu_train:.6f}")
    print(f"  Std Dev (σ):     {sigma_train:.6f}")
    print(f"  Min:             {errors_train.min():.6f}")
    print(f"  Max:             {errors_train.max():.6f}")
    print(f"  Median:          {np.median(errors_train):.6f}")
    print(f"\nGlobal Threshold (μ + {N_SIGMA}σ): {global_threshold:.6f}")
    
    # Expected false positive rate (assuming normal distribution)
    expected_fpr = (errors_train > global_threshold).sum() / len(errors_train)
    print(f"Expected FPR on train: {expected_fpr:.4f} ({expected_fpr*100:.2f}%)")
    
    # Check anomaly rate in validation
    anomalies_val = errors_val > global_threshold
    n_anomalies_val = anomalies_val.sum()
    anomaly_rate_val = (n_anomalies_val / len(errors_val)) * 100
    print(f"\nValidation anomaly rate: {n_anomalies_val}/{len(errors_val)} ({anomaly_rate_val:.2f}%)")
    
    
    # ========================================================================
    # VISUALIZE TRAINING AND PER-SENSOR ERRORS
    # ========================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training_history(history, f'{OUTPUT_DIR}/training_history_{CLUSTER}.png')
    
    # Plot reconstruction errors
    plt.figure(figsize=(14, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors_train, bins=50, alpha=0.7, label='Train', edgecolor='black')
    plt.hist(errors_val, bins=50, alpha=0.7, label='Validation', edgecolor='black')
    plt.axvline(global_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Global Threshold: {global_threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Global Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(errors_train)), errors_train, alpha=0.5, s=10, label='Train')
    plt.scatter(range(len(errors_val)), errors_val, alpha=0.5, s=10, label='Validation')
    plt.axhline(global_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Global Threshold: {global_threshold:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Global Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/global_reconstruction_errors_{CLUSTER}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot per-sensor error distributions
    plt.figure(figsize=(15, 10))
    
    n_sensors = len(sensor_names)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    for i, sensor_name in enumerate(sensor_names):
        plt.subplot(n_rows, n_cols, i + 1)
        sensor_errors = per_sensor_errors_val[:, i]
        threshold = per_sensor_thresholds[sensor_name]
        
        plt.hist(sensor_errors, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Frequency')
        plt.title(f'Sensor: {sensor_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/per_sensor_error_distributions_{CLUSTER}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot per-sensor thresholds comparison
    plt.figure(figsize=(12, 6))
    
    sensors = list(per_sensor_thresholds.keys())
    thresholds = list(per_sensor_thresholds.values())
    
    bars = plt.bar(sensors, thresholds, alpha=0.7, edgecolor='black')
    plt.axhline(y=global_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Global Threshold: {global_threshold:.4f}')
    
    # Color bars based on threshold relative to global threshold
    for j, (bar, threshold) in enumerate(zip(bars, thresholds)):
        if threshold > global_threshold * 1.2:
            bar.set_color('orange')
        elif threshold > global_threshold * 1.1:
            bar.set_color('yellow')
        else:
            bar.set_color('green')
    
    plt.xlabel('Sensor', fontsize=12)
    plt.ylabel('Threshold Value', fontsize=12)
    plt.title('Per-Sensor vs Global Thresholds', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for j, (sensor, threshold) in enumerate(zip(sensors, thresholds)):
        plt.text(j, threshold + 0.0005, f'{threshold:.4f}', 
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/threshold_comparison_{CLUSTER}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # SAVE MODEL, SCALERS, AND METADATA
    # ========================================================================
    
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    # Save model
    autoencoder.model.save(f'{OUTPUT_DIR}/bilstm_autoencoder_{CLUSTER}.h5')
    print(f"✓ Model saved: {OUTPUT_DIR}/bilstm_autoencoder_{CLUSTER}.h5")
    
    # Save scalers
    joblib.dump(scalers, f'{OUTPUT_DIR}/scalers_{CLUSTER}.pkl')
    print(f"✓ Scalers saved: {OUTPUT_DIR}/scalers_{CLUSTER}.pkl")
    
    # Save per-sensor errors for validation set (for analysis)
    np.save(f'{OUTPUT_DIR}/per_sensor_val_errors_{CLUSTER}.npy', per_sensor_errors_val)
    print(f"✓ Per-sensor validation errors saved: {OUTPUT_DIR}/per_sensor_val_errors_{CLUSTER}.npy")
    
    # Save metadata with per-sensor thresholds
    metadata = {
        'model_type': 'bilstm_autoencoder_fused',
        'sensor_names': sensor_names,
        'n_sensors': n_features,
        'sequence_length': SEQUENCE_LENGTH,
        'hyperparameters': {
            'hidden_units': HIDDEN_UNITS,
            'encoding_dim': ENCODING_DIM,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'patience_factor': PATIENCE_FACTOR,
            'calculated_patience': calculated_patience
        },
        'global_threshold': {
            'value': float(global_threshold),
            'method': f'mu_plus_{N_SIGMA}_sigma',
            'mu': float(mu_train),
            'sigma': float(sigma_train),
            'n_sigma': N_SIGMA
        },
        'per_sensor_thresholds': per_sensor_thresholds,
        'per_sensor_stats': per_sensor_stats,
        'data_split': {
            'train_samples': len(X_train_scaled),
            'val_samples': len(X_val_scaled),
            'train_ratio': TRAIN_RATIO
        },
        'training': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            'epochs_trained': len(history.history['loss']),
            'best_epoch': int(np.argmin(history.history['val_loss']) + 1)
        },
        'error_statistics': {
            'train': {
                'mean': float(errors_train.mean()),
                'std': float(errors_train.std()),
                'min': float(errors_train.min()),
                'max': float(errors_train.max()),
                'median': float(np.median(errors_train))
            },
            'validation': {
                'mean': float(errors_val.mean()),
                'std': float(errors_val.std()),
                'min': float(errors_val.min()),
                'max': float(errors_val.max()),
                'median': float(np.median(errors_val))
            }
        },
        'anomaly_rates': {
            'train_expected_fpr': float(expected_fpr),
            'validation_anomaly_rate': float(anomaly_rate_val),
            'validation_anomalies': int(n_anomalies_val)
        },
        'environment': {
            'tensorflow_version': tf.__version__,
            'cluster': CLUSTER,
            'random_seed': 42
        }
    }
    
    with open(f'{OUTPUT_DIR}/metadata_{CLUSTER}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {OUTPUT_DIR}/metadata_{CLUSTER}.json")
    
    # Save reconstruction errors for analysis
    np.save(f'{OUTPUT_DIR}/train_errors_{CLUSTER}.npy', errors_train)
    np.save(f'{OUTPUT_DIR}/val_errors_{CLUSTER}.npy', errors_val)
    print(f"✓ Reconstruction errors saved as numpy arrays")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - INCLUDING PER-SENSOR THRESHOLDING!")
    print("="*60)
    print(f"\nModel artifacts saved in: {OUTPUT_DIR}/")
    print("Files created:")
    print("  - bilstm_autoencoder.h5 (trained model)")
    print("  - scalers.pkl (sensor scalers)")
    print("  - metadata.json (configuration and results)")
    print("  - train_errors.npy, val_errors.npy")
    print("  - per_sensor_val_errors.npy (for localization)")
    print("  - training_history.png")
    print("  - global_reconstruction_errors.png")
    print("  - per_sensor_error_distributions.png")
    print("  - threshold_comparison.png")
    print(f"\nModel Summary:")
    print(f"  Sensors: {n_features} ({', '.join(sensor_names)})")
    print(f"  Architecture: BiLSTM({HIDDEN_UNITS}) → Dense({ENCODING_DIM}) → Repeat → BiLSTM({HIDDEN_UNITS}) → Output")
    print(f"  Global Threshold: {global_threshold:.6f} (μ + {N_SIGMA}σ)")
    print(f"\nPer-Sensor Thresholds:")
    for sensor, threshold in per_sensor_thresholds.items():
        print(f"  {sensor}: {threshold:.6f}")