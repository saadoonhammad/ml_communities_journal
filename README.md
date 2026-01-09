
# Anomaly detection for Communities of Interest in Internet of Things using Deep Autoencoders

A deep learning framework for detecting collective anomalies in IoT temperature sensor data using autoencoder architectures with hierarchical clustering.

## Overview

This repository contains the implementation of a collective anomaly detection methodology designed to identify abnormal sensor behaviour patterns in temperature time series data from a network of meteorological stations in the province of Castelló. The approach integrates hierarchical clustering with three autoencoder architectures (BiLSTM, LSTM, and MLP) to detect collective anomalies such as sustained temperature drops indicating sensor malfunctions.

### Key Features

- **Hierarchical Clustering**: Groups sensors based on temporal, spatial, and elevation similarities (4 clusters: C1-C4)
- **Multiple Autoencoder Architectures**: MLP, LSTM and BiLSTM implementations
- **Comprehensive Feature Engineering**: 13-feature pipeline including temporal encodings, statistical features.
- **Rigorous Hyperparameter Optimization**: Bayesian optimization with expanding window cross-validation


## Repository Structure

This repository contains the implementation of three autoencoder architectures for IoT temperature sensor anomaly detection across four clusters (C1-C4).

### BiLSTM (`/bilstm`)
- **`bilstm_hpo.py`** - Bayesian hyperparameter optimization for BiLSTM autoencoder models
- **`bi_lstm_final_train.py`** - Final training of BiLSTM models using optimized hyperparameters
- **`bi_lstm_test.py`** - Model evaluation and generalizability testing across clusters

### LSTM (`/lstm`)
- **`lstm_hpo.py`** - Bayesian hyperparameter optimization for LSTM autoencoder models
- **`lstm_final_train.py`** - Final training of LSTM models using optimized hyperparameters
- **`lstm_test.py`** - Model evaluation and generalizability testing across clusters

### MLP (`/mlp`)
- **`mlp_hpo.py`** - Bayesian hyperparameter optimization for MLP autoencoder models
- **`mlp_final_train.py`** - Final training of MLP models using optimized hyperparameters
- **`mlp_test.py`** - Model evaluation and generalizability testing across clusters

Each architecture folder contains trained models and test results organized by cluster (C1-C4). 

## Architecture

### Autoencoder Models

1. **MLP Autoencoder**: Dense layers for baseline comparison and computational efficiency
2. **BiLSTM Autoencoder**: Bidirectional LSTM layers capture temporal dependencies in both directions
3. **LSTM Autoencoder**: Unidirectional LSTM for sequential pattern learning

### Feature Engineering Pipeline

The system extracts 13 features from raw temperature readings:

- **Raw Temperature**: Original sensor value
- **Cyclical Temporal Encodings**: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- **Derivative Features**: Velocity (1st derivative), Acceleration (2nd derivative), Energy
- **Volatility Measures**: For detecting stuck or erratic sensors
- **Statistical Aggregations**: Rolling mean, standard deviation, range

## Methodology

### Data Processing and Clustering

- **Frequency**: 10-minute intervals
- **Sequence Length**: 24
- **Clusters**: 4 clusters formed through hierarchical clustering

### Cross-Validation

Expanding window cross-validation (5 folds) to preserve temporal order and prevent data leakage:

```
Fold 1: Train [0:20%]    → Validate [20:40%]
Fold 2: Train [0:40%]    → Validate [40:60%]
Fold 3: Train [0:60%]    → Validate [60:80%]
Fold 4: Train [0:80%]    → Validate [80:100%]
Fold 5: Train [0:100%]   → Validate on holdout
```

### Hyperparameter Optimisation

Bayesian optimisation (35 evaluations) using Gaussian Process with:

- Constrained search spaces forcing regularisation
- Tight bottleneck architectures
- Prevention of boundary-hitting behaviour
- Optimisation for anomaly detection (not just reconstruction)

## Results

### Performance Metrics for Anomaly Detection

Models evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Specificity, AUC-ROC, PR-AUC

**Threshold Strategy**: Statistical thresholds (μ + 3σ) proved more practical than ROC-based optimization

### Hardware Used

- **Training**: NVIDIA GPU (tested on RTX 5090)

## Reproducibility

All experiments use fixed random seeds:

```python
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()
```

Comprehensive metadata and scalers are saved for traceability across experiments.

## Key Insights

### Data Leakage Prevention

- Expanding window cross-validation maintains chronological order
- Purge gaps between training and validation sets
- Proper sequence splitting to avoid temporal leakage

### Optimisation Strategy

**Critical Discovery**: Optimising for reconstruction loss minimization is counterproductive for anomaly detection. Models that reconstruct everything perfectly fail to distinguish normal from anomalous patterns.

**Solution**: Optimize for anomaly detection performance metrics (F1-score, Recall) rather than reconstruction error alone.

### Contextual Features

Statistical thresholds and contextual features are essential for detecting collective anomalies that appear normal in isolation.

## Future Work

- [ ] Complete generalizability evaluation framework
- [ ] Feature importance analysis using permutation methods
- [ ] Extended validation on diverse building types
- [ ] Real-time monitoring dashboard
