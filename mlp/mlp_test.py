# ============================================================================
# BATCH ANOMALY DETECTION TESTING - MULTIPLE SENSORS
# ============================================================================
"""
Batch testing script for MLP Autoencoder anomaly detection across multiple sensors.

This script:
1. Loads a trained MLP autoencoder model
2. Tests the model on multiple sensor test files
3. Generates comprehensive metrics and visualizations for each sensor
4. Creates a summary CSV with all metrics
5. Generates comparison visualizations across sensors

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import glob
from pathlib import Path
import traceback
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU(s) detected: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - running on CPU")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
CLUSTER = 'C3'
output_cluster= 'c3'
TEST_DATA_FOLDER = f'1. Datasets/Test Sets/{CLUSTER}/'  # Folder containing all test CSV files
# MODEL_DIR = 'trained_model'
MODEL_DIR = f'2. Algorithms/MLP/trained_model/{output_cluster}'
OUTPUT_DIR = f'test_results_batch_{output_cluster}'

# Create output directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_METRICS_FILE = os.path.join(OUTPUT_DIR, 'summary_metrics.csv')


# ============================================================================
# HELPER FUNCTIONS - FEATURE ENGINEERING
# ============================================================================

def create_sequences(data, sequence_length):
    """Create sliding window sequences"""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


def add_temporal_features(df):
    """Add contextual cyclical temporal features (Sine/Cosine)."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    
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
    """Add inverse volatility features to detect flat-line sensors"""
    df = df.copy()
    
    rolling_std = df[column].rolling(window=10).std()
    rolling_std.fillna(method='bfill', inplace=True)
    
    epsilon = 1e-3 
    df['static_sensor_alert'] = 1.0 / (rolling_std + epsilon)
    df['static_sensor_alert'] = np.log1p(df['static_sensor_alert'])
    
    return df


def add_statistical_features(sequences):
    """Add statistical features to each sequence"""
    n_samples = sequences.shape[0]
    seq_len = sequences.shape[1]
    
    stats_features = []
    
    for seq in sequences:
        temp_values = seq[:, 0]
        
        mean_val = np.mean(temp_values)
        std_val = np.std(temp_values)
        range_val = np.max(temp_values) - np.min(temp_values)
        
        stats = np.column_stack([
            np.full(seq_len, mean_val),
            np.full(seq_len, std_val),
            np.full(seq_len, range_val)
        ])
        
        stats_features.append(stats)
    
    stats_features = np.array(stats_features)
    enhanced_sequences = np.concatenate([sequences, stats_features], axis=2)
    
    return enhanced_sequences


def prepare_test_data_with_features(df, sequence_length, feature_mode='all'):
    """
    Prepare TEST data with the SAME feature configuration as training
    
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
        labels: numpy array of anomaly labels (aligned with sequences)
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    if df['temp_value'].isna().any():
        df['temp_value'].fillna(method='ffill', inplace=True)
    
    # Extract labels BEFORE creating sequences
    all_labels = df['anomaly_label'].values
    
    if feature_mode == 'raw':
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        feature_names = ['temp_value']
        
    elif feature_mode == 'temporal':
        df = add_temporal_features(df)
        feature_cols = ['temp_value', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        feature_names = feature_cols
        
    elif feature_mode == 'statistical':
        data = df['temp_value'].values.reshape(-1, 1)
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = ['temp_value', 'mean', 'std', 'range']
        
    elif feature_mode == 'derivative':
        df = add_derivative_features(df)
        feature_cols = ['temp_value', 'energy', 'velocity', 'acceleration']
        data = df[feature_cols].values
        sequences = create_sequences(data, sequence_length)
        sequences = add_statistical_features(sequences)
        feature_names = feature_cols + ['mean', 'std', 'range']
        
    elif feature_mode == 'all':
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
    
    # Align labels with sequences
    sequence_labels = all_labels[sequence_length:sequence_length + len(sequences)]
    
    return sequences, sequence_labels


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, save_path, sensor_id=''):
    """Plot confusion matrix with sensor ID in title"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    title = f'Confusion Matrix - {sensor_id}' if sensor_id else 'Confusion Matrix'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path, sensor_id=''):
    """Plot ROC curve with sensor ID in title"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f'ROC Curve - {sensor_id}' if sensor_id else 'ROC Curve'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(precision, recall, save_path, sensor_id=''):
    """Plot Precision-Recall curve with AUC score and sensor ID in title"""
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = f'Precision-Recall Curve - {sensor_id}' if sensor_id else 'Precision-Recall Curve'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return pr_auc


def plot_reconstruction_errors_with_labels(errors, labels, threshold, save_path, sensor_id=''):
    """Plot reconstruction errors colored by true labels with sensor ID in title"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Separate errors by label
    errors_normal = errors[labels == 0]
    errors_anomaly = errors[labels == 1]
    
    # Histogram
    ax1.hist(errors_normal, bins=50, alpha=0.7, label='Normal', 
             edgecolor='black', color='blue')
    ax1.hist(errors_anomaly, bins=50, alpha=0.7, label='Anomaly', 
             edgecolor='black', color='red')
    ax1.axvline(threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    ax1.set_xlabel('Reconstruction Error (MSE)')
    ax1.set_ylabel('Frequency')
    title = f'Reconstruction Error Distribution - {sensor_id}' if sensor_id else 'Reconstruction Error Distribution'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot over time
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    ax2.scatter(normal_indices, errors[normal_indices], alpha=0.6, s=20, 
                label='Normal', color='blue')
    ax2.scatter(anomaly_indices, errors[anomaly_indices], alpha=0.8, s=30, 
                label='Anomaly', color='red', marker='x')
    ax2.axhline(threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('Reconstruction Errors Over Time with True Labels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# SINGLE FILE TESTING FUNCTION
# ============================================================================

def test_single_file(test_csv_path, model, scalers, metadata, threshold, output_subdir):
    """
    Test a single CSV file and return metrics
    
    Args:
        test_csv_path: path to test CSV file
        model: loaded Keras model
        scalers: list of fitted scalers
        metadata: metadata dict from training
        threshold: anomaly detection threshold
        output_subdir: directory to save outputs for this sensor
        
    Returns:
        dict: metrics for this test file, or None if failed
    """
    
    sensor_id = Path(test_csv_path).stem
    print(f"\n{'='*80}")
    print(f"TESTING: {sensor_id}")
    print(f"{'='*80}")
    
    # Create output subdirectory
    os.makedirs(output_subdir, exist_ok=True)
    
    # ========================================================================
    # LOAD TEST DATA
    # ========================================================================
    
    try:
        df_test = pd.read_csv(test_csv_path)
        print(f"✓ Loaded {len(df_test)} samples")
        
        # Validate required columns
        if 'anomaly_label' not in df_test.columns:
            raise ValueError("'anomaly_label' column not found in test data")
        if 'temp_value' not in df_test.columns:
            raise ValueError("'temp_value' column not found in test data")
        if 'timestamp' not in df_test.columns:
            raise ValueError("'timestamp' column not found in test data")
            
        print(f"  Anomalies: {df_test['anomaly_label'].sum()} "
              f"({df_test['anomaly_label'].sum()/len(df_test)*100:.2f}%)")
        
    except Exception as e:
        print(f"✗ ERROR loading {test_csv_path}: {str(e)}")
        traceback.print_exc()
        return None
    
    # ========================================================================
    # PREPARE TEST SEQUENCES
    # ========================================================================
    
    try:
        FEATURE_MODE = metadata['feature_mode']
        SEQUENCE_LENGTH = metadata['sequence_length']
        n_features = metadata['n_features']
        
        X_test_raw, y_test = prepare_test_data_with_features(
            df_test, SEQUENCE_LENGTH, feature_mode=FEATURE_MODE
        )
        
        # Verify feature count
        if X_test_raw.shape[2] != n_features:
            raise ValueError(f"Feature mismatch! Training: {n_features}, Test: {X_test_raw.shape[2]}")
        
        print(f"✓ Created {X_test_raw.shape[0]} sequences with {X_test_raw.shape[2]} features")
            
    except Exception as e:
        print(f"✗ ERROR preparing features: {str(e)}")
        traceback.print_exc()
        return None
    
    # ========================================================================
    # APPLY SCALING
    # ========================================================================
    
    try:
        X_test_scaled = X_test_raw.copy()
        
        for i in range(n_features):
            X_test_scaled[:, :, i] = scalers[i].transform(
                X_test_raw[:, :, i].reshape(-1, 1)
            ).reshape(X_test_raw.shape[0], SEQUENCE_LENGTH)
        
        print(f"✓ Scaled {X_test_scaled.shape[0]} sequences")
        
    except Exception as e:
        print(f"✗ ERROR scaling data: {str(e)}")
        traceback.print_exc()
        return None
    
    # ========================================================================
    # GET PREDICTIONS
    # ========================================================================
    
    try:
        reconstructions = model.predict(X_test_scaled, verbose=0)
        errors_test = np.mean(np.square(X_test_scaled - reconstructions), axis=(1, 2))
        
        print(f"✓ Reconstruction errors computed")
        print(f"  Mean: {errors_test.mean():.6f}, Std: {errors_test.std():.6f}")
        
    except Exception as e:
        print(f"✗ ERROR computing predictions: {str(e)}")
        traceback.print_exc()
        return None
    
    # ========================================================================
    # CLASSIFY ANOMALIES
    # ========================================================================
    
    y_pred = (errors_test > threshold).astype(int)
    
    print(f"  Predicted anomalies: {y_pred.sum()} / {len(y_pred)} "
          f"({y_pred.sum()/len(y_pred)*100:.2f}%)")
    
    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================
    
    try:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y_test, errors_test)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, errors_test)
        pr_auc = auc(recall_curve, precision_curve)
        
        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY - {sensor_id}")
        print(f"{'='*60}")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-Score:    {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"PR-AUC:      {pr_auc:.4f}")
        
    except Exception as e:
        print(f"✗ ERROR calculating metrics: {str(e)}")
        traceback.print_exc()
        return None
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    
    try:
        print(f"\nGenerating visualizations...")
        
        plot_confusion_matrix(cm, 
            os.path.join(output_subdir, 'confusion_matrix.png'), sensor_id)
        
        plot_roc_curve(fpr, tpr, roc_auc, 
            os.path.join(output_subdir, 'roc_curve.png'), sensor_id)
        
        plot_precision_recall_curve(precision_curve, recall_curve, 
            os.path.join(output_subdir, 'precision_recall_curve.png'), sensor_id)
        
        plot_reconstruction_errors_with_labels(errors_test, y_test, threshold,
            os.path.join(output_subdir, 'reconstruction_errors_labeled.png'), sensor_id)
        
        print(f"✓ All visualizations saved")
            
    except Exception as e:
        print(f"✗ ERROR generating plots: {str(e)}")
        traceback.print_exc()
    
    # ========================================================================
    # SAVE INDIVIDUAL PREDICTIONS CSV
    # ========================================================================
    
    try:
        results_df = pd.DataFrame({
            'sample_index': range(len(y_test)),
            'reconstruction_error': errors_test,
            'true_label': y_test,
            'predicted_label': y_pred,
            'is_correct': (y_test == y_pred).astype(int)
        })
        predictions_path = os.path.join(output_subdir, 'predictions.csv')
        results_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved: {predictions_path}")
        
    except Exception as e:
        print(f"✗ ERROR saving predictions: {str(e)}")
        traceback.print_exc()
    
    # ========================================================================
    # SAVE INDIVIDUAL TEST RESULTS JSON
    # ========================================================================
    
    try:
        test_results = {
            'sensor_id': sensor_id,
            'test_file': test_csv_path,
            'model_type': 'MLP Autoencoder',
            'feature_mode': metadata['feature_mode'],
            'sequence_length': SEQUENCE_LENGTH,
            'threshold': float(threshold),
            'test_samples': len(y_test),
            'true_anomalies': int(y_test.sum()),
            'predicted_anomalies': int(y_pred.sum()),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc)
            },
            'reconstruction_errors': {
                'mean': float(errors_test.mean()),
                'std': float(errors_test.std()),
                'min': float(errors_test.min()),
                'max': float(errors_test.max())
            }
        }
        
        json_path = os.path.join(output_subdir, 'test_results.json')
        with open(json_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"✓ Test results JSON saved: {json_path}")
        
    except Exception as e:
        print(f"✗ ERROR saving JSON: {str(e)}")
        traceback.print_exc()
    
    # ========================================================================
    # RETURN METRICS DICT FOR SUMMARY CSV
    # ========================================================================
    
    return {
        'sensor_id': sensor_id,
        'test_file': test_csv_path,
        'n_samples': len(y_test),
        'n_anomalies': int(y_test.sum()),
        'n_predicted_anomalies': int(y_pred.sum()),
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'recon_error_mean': float(errors_test.mean()),
        'recon_error_std': float(errors_test.std()),
        'recon_error_min': float(errors_test.min()),
        'recon_error_max': float(errors_test.max()),
        'threshold': float(threshold)
    }


# ============================================================================
# MAIN EXECUTION - BATCH TESTING
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("BATCH ANOMALY DETECTION TESTING - MULTIPLE SENSORS")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD MODEL AND METADATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: LOADING MODEL AND METADATA")
    print("="*80)
    
    try:
        # Load metadata
        with open(f'{MODEL_DIR}/metadata_{output_cluster}.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded:")
        print(f"  Feature mode: {metadata['feature_mode']}")
        print(f"  Sequence length: {metadata['sequence_length']}")
        print(f"  Number of features: {metadata['n_features']}")
        print(f"  Threshold: {metadata['threshold']:.6f}")
        
        THRESHOLD = metadata['threshold']
        
    except Exception as e:
        print(f"✗ ERROR loading metadata: {str(e)}")
        exit(1)
    
    try:
        # Load scalers
        scalers = joblib.load(f'{MODEL_DIR}/scalers_{output_cluster}.pkl')
        print(f"✓ Loaded {len(scalers)} scalers")
        
    except Exception as e:
        print(f"✗ ERROR loading scalers: {str(e)}")
        exit(1)
    
    try:
        # Load model
        model = load_model(f'{MODEL_DIR}/mlp_autoencoder_{output_cluster}.h5', compile=False)
        print(f"✓ Model loaded from: {MODEL_DIR}/mlp_autoencoder.h5")
        
        # Recompile
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )
        print("✓ Model recompiled successfully")
        
    except Exception as e:
        print(f"✗ ERROR loading model: {str(e)}")
        exit(1)
    
    # ========================================================================
    # STEP 2: FIND ALL TEST FILES
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: FINDING TEST FILES")
    print("="*80)
    
    # Find all CSV files in the test data folder
    test_files = sorted(glob.glob(os.path.join(TEST_DATA_FOLDER, '*.csv')))
    
    if len(test_files) == 0:
        print(f"✗ ERROR: No CSV files found in {TEST_DATA_FOLDER}")
        exit(1)
    
    print(f"✓ Found {len(test_files)} test files:")
    for i, filepath in enumerate(test_files, 1):
        filename = Path(filepath).name
        print(f"  {i}. {filename}")
    
    # ========================================================================
    # STEP 3: PROCESS EACH TEST FILE
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: PROCESSING TEST FILES")
    print("="*80)
    
    all_metrics = []
    successful_tests = 0
    failed_tests = 0
    
    for test_file in test_files:
        sensor_id = Path(test_file).stem
        output_subdir = os.path.join(OUTPUT_DIR, sensor_id)
        
        try:
            metrics = test_single_file(
                test_csv_path=test_file,
                model=model,
                scalers=scalers,
                metadata=metadata,
                threshold=THRESHOLD,
                output_subdir=output_subdir
            )
            
            if metrics is not None:
                all_metrics.append(metrics)
                successful_tests += 1
                print(f"✓ Successfully processed: {sensor_id}")
            else:
                failed_tests += 1
                print(f"✗ Failed to process: {sensor_id}")
                
        except Exception as e:
            failed_tests += 1
            print(f"✗ UNEXPECTED ERROR processing {sensor_id}: {str(e)}")
            traceback.print_exc()
            continue
    
    # ========================================================================
    # STEP 4: SAVE SUMMARY METRICS CSV
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: SAVING SUMMARY METRICS")
    print("="*80)
    
    if len(all_metrics) > 0:
        try:
            summary_df = pd.DataFrame(all_metrics)
            
            # Reorder columns for better readability
            column_order = [
                'sensor_id', 'test_file', 'n_samples', 'n_anomalies', 
                'n_predicted_anomalies', 'TP', 'FP', 'TN', 'FN',
                'accuracy', 'precision', 'recall', 'f1_score', 
                'specificity', 'roc_auc', 'pr_auc',
                'recon_error_mean', 'recon_error_std', 
                'recon_error_min', 'recon_error_max', 'threshold'
            ]
            
            summary_df = summary_df[column_order]
            summary_df.to_csv(SUMMARY_METRICS_FILE, index=False)
            
            print(f"✓ Summary metrics saved: {SUMMARY_METRICS_FILE}")
            print(f"\nSummary Statistics:")
            print(f"  Total files processed: {len(test_files)}")
            print(f"  Successful: {successful_tests}")
            print(f"  Failed: {failed_tests}")
            print(f"\nAverage Metrics Across All Sensors:")
            print(f"  Accuracy:    {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}")
            print(f"  Precision:   {summary_df['precision'].mean():.4f} ± {summary_df['precision'].std():.4f}")
            print(f"  Recall:      {summary_df['recall'].mean():.4f} ± {summary_df['recall'].std():.4f}")
            print(f"  F1-Score:    {summary_df['f1_score'].mean():.4f} ± {summary_df['f1_score'].std():.4f}")
            print(f"  Specificity: {summary_df['specificity'].mean():.4f} ± {summary_df['specificity'].std():.4f}")
            print(f"  ROC-AUC:     {summary_df['roc_auc'].mean():.4f} ± {summary_df['roc_auc'].std():.4f}")
            print(f"  PR-AUC:      {summary_df['pr_auc'].mean():.4f} ± {summary_df['pr_auc'].std():.4f}")
            
        except Exception as e:
            print(f"✗ ERROR saving summary metrics: {str(e)}")
            traceback.print_exc()
    else:
        print("✗ No successful tests - cannot create summary metrics CSV")
    
    # ========================================================================
    # STEP 5: GENERATE COMPARISON VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    if len(all_metrics) > 1:
        try:
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            sensor_ids = summary_df['sensor_id'].values
            x_pos = np.arange(len(sensor_ids))
            
            # Plot 1: Accuracy, Precision, Recall, F1
            ax1 = axes[0, 0]
            width = 0.2
            ax1.bar(x_pos - 1.5*width, summary_df['accuracy'], width, label='Accuracy', alpha=0.8)
            ax1.bar(x_pos - 0.5*width, summary_df['precision'], width, label='Precision', alpha=0.8)
            ax1.bar(x_pos + 0.5*width, summary_df['recall'], width, label='Recall', alpha=0.8)
            ax1.bar(x_pos + 1.5*width, summary_df['f1_score'], width, label='F1-Score', alpha=0.8)
            ax1.set_xlabel('Sensor ID')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(sensor_ids, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1.1])
            
            # Plot 2: ROC-AUC and PR-AUC
            ax2 = axes[0, 1]
            width = 0.35
            ax2.bar(x_pos - width/2, summary_df['roc_auc'], width, label='ROC-AUC', alpha=0.8)
            ax2.bar(x_pos + width/2, summary_df['pr_auc'], width, label='PR-AUC', alpha=0.8)
            ax2.set_xlabel('Sensor ID')
            ax2.set_ylabel('AUC Score')
            ax2.set_title('AUC Metrics Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(sensor_ids, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.1])
            
            # Plot 3: Confusion Matrix Components
            ax3 = axes[1, 0]
            width = 0.2
            ax3.bar(x_pos - 1.5*width, summary_df['TP'], width, label='TP', alpha=0.8, color='green')
            ax3.bar(x_pos - 0.5*width, summary_df['FP'], width, label='FP', alpha=0.8, color='orange')
            ax3.bar(x_pos + 0.5*width, summary_df['TN'], width, label='TN', alpha=0.8, color='blue')
            ax3.bar(x_pos + 1.5*width, summary_df['FN'], width, label='FN', alpha=0.8, color='red')
            ax3.set_xlabel('Sensor ID')
            ax3.set_ylabel('Count')
            ax3.set_title('Confusion Matrix Components')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(sensor_ids, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Reconstruction Error Statistics
            ax4 = axes[1, 1]
            ax4.bar(x_pos, summary_df['recon_error_mean'], 
                   yerr=summary_df['recon_error_std'], 
                   alpha=0.7, capsize=5, color='purple')
            ax4.axhline(y=THRESHOLD, color='red', linestyle='--', 
                       linewidth=2, label=f'Threshold: {THRESHOLD:.4f}')
            ax4.set_xlabel('Sensor ID')
            ax4.set_ylabel('Reconstruction Error (MSE)')
            ax4.set_title('Mean Reconstruction Error (± Std Dev)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(sensor_ids, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_plot_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
            plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Comparison plot saved: {comparison_plot_path}")
            
        except Exception as e:
            print(f"✗ ERROR generating comparison visualizations: {str(e)}")
            traceback.print_exc()
    else:
        print("⚠ Not enough data for comparison plots (need at least 2 sensors)")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("BATCH TESTING COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  ✓ {SUMMARY_METRICS_FILE} (all metrics in one CSV)")
    
    if len(all_metrics) > 0:
        for metrics in all_metrics:
            sensor_id = metrics['sensor_id']
            print(f"  ✓ {sensor_id}/")
            print(f"      - confusion_matrix.png")
            print(f"      - roc_curve.png")
            print(f"      - precision_recall_curve.png")
            print(f"      - reconstruction_errors_labeled.png")
            print(f"      - predictions.csv")
            print(f"      - test_results.json")
    
    if len(all_metrics) > 1:
        print(f"  ✓ metrics_comparison.png")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total test files found: {len(test_files)}")
    print(f"Successfully processed: {successful_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests > 0:
        print(f"\n⚠ Warning: {failed_tests} file(s) failed processing. Check logs above for details.")
    
    print(f"\n{'='*80}")