
# Community-Based Model Sharing and Generalisation: Anomaly Detection IoT Temperature Sensors Network

A deep learning framework for detecting collective anomalies in IoT temperature sensor data using autoencoder architectures in Communities of Interest (CoIs).

## Overview

This repository contains the implementation of a collective anomaly detection methodology designed to identify abnormal sensor behaviour patterns in temperature time series data from a network of meteorological stations in the province of Castelló. The approach integrates hierarchical clustering to form Communities of Interest (CoIs). Three autoencoder architectures (BiLSTM, LSTM, and MLP) are trained to detect collective anomalies such as sustained temperature drops indicating sensor malfunctions.

### Key Features

- **Hierarchical Clustering**: Groups sensors based on temporal, spatial, and elevation similarities (4 clusters: C1-C4)
- **Multiple Autoencoder Architectures**: MLP, LSTM and BiLSTM implementations
- **Comprehensive Feature Engineering**: 13-feature pipeline including temporal encodings, statistical features.
- **Rigorous Hyperparameter Optimisation**: Bayesian Hyperband Optimisation with expanding window cross-validation
- **Comparison**: Comparison with a global baseline to show the effectiveness of CoIs

## Architectures

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

Expanding window cross‑validation (5 folds) with purge gap
- To preserve temporal order and prevent data leakage in time series, we use an expanding window strategy with a mandatory purge gap between training and validation sets.
- The purge gap length is 2 × sequence_length (default 48 time steps), removing any autocorrelated influence from the training period.
- The validation set size is fixed and is automatically computed to maximise data usage while ensuring a feasible step between folds.
- The training set grows with each fold, but the validation window always moves forward without overlap.

## Hyperparameter Optimisation

Bayesian Hyperband Optimisation (BOHB) with 60 evaluations. We combine the Tree‑structured Parzen Estimator (TPE) with Hyperband pruning for efficient exploration of the hyperparameter space.

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

### Contextual Features

Statistical thresholds and contextual features are essential for detecting collective anomalies that appear normal in isolation.

## Future Work

-  Complete generalisability evaluation framework
-  Feature importance analysis using permutation methods
-  Implementation of the trained models on resource-constrained devices such as microcontrollers
