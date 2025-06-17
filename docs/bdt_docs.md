# # Document written by Claude.ai

# BDT.py - Boosted Decision Tree Framework Documentation

## Overview

This module provides a comprehensive framework for training and deploying Boosted Decision Tree (BDT) models using XGBoost. It includes classes for classification, regression, data preprocessing, and model evaluation for waveform analysis and classification tasks.

## Utility Functions

### `split_train_test_dataset(path_to_data, frac_test=0.2)`

**Purpose**: Splits labeled datasets into training/validation and testing sets by class.

**Parameters**:

- `path_to_data`: Path to directory containing CSV files
- `frac_test`: Fraction of data to use for testing (default: 0.2)

**Process**: Creates `train_valid/` and `test/` subdirectories with class-balanced splits.

### `confusionMatrix(truth_Series, pred_Series, output_path, figname)`

**Purpose**: Generates and saves confusion matrix visualization.

### `corr2dplot(truth_df, pred_df, output_path, classname)`

**Purpose**: Creates 2D correlation plots comparing predicted vs true values for metrics.

### `compare_metrics(data_df, output_path)`

**Purpose**: Generates histogram comparisons between predicted and true metric values.

---

## Core Classes

## 1. BDT_Classifier

**Purpose**: Trains XGBoost classification models to categorize samples into predefined classes based on computed metrics.

### Constructor

```python
def __init__(self, path_to_data=None, output_path=None):
```

### Key Attributes

- **`input_columns`**: `['integral_R', 'max_deviation']` - Features used for classification
- **`class_map`**: Maps class names ('c1', 'c2', 'c3', 'c4') to numeric labels (0, 1, 2, 3)
- **`train_df`**, **`test_df`**: Training and testing datasets
- **`classifier_model`**: XGBoost classifier instance

### Methods

#### `tune_hyperparameters()`

**Purpose**: Performs randomized search for optimal hyperparameters.

**Search Space**:

- `n_estimators`: 100-200
- `max_depth`: 15-20
- `learning_rate`: 0.4-0.7
- `min_child_weight`: 15-20
- `subsample`: 0.8-0.9
- GPU acceleration enabled

#### `train(params={})`

**Purpose**: Trains the classification model with provided parameters.

**Process**:

1. Splits training data into train/validation (80/20)
2. Trains XGBoost classifier with early stopping
3. Evaluates on test dataset
4. Saves model if accuracy ≥ 99.85%
5. Generates confusion matrix for evaluation

---

## 2. BDT_Regressor

**Purpose**: Trains XGBoost regression models to predict continuous metrics from waveform fit parameters.

### Constructor

```python
def __init__(self, path_to_data=None, output_path=None):
```

### Key Attributes

- **`input_columns`**: `['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']` - Fit parameters as features
- **`output_columns`**: `['integral_R', 'max_deviation']` - Target metrics to predict
- **`regressor_model`**: XGBoost regressor instance

### Methods

#### `tune_hyperparameters()`

**Purpose**: Optimizes regression model hyperparameters using randomized search.

#### `train(params={})`

**Purpose**: Trains regression model and evaluates performance.

**Process**:

1. Trains model to predict `integral_R` and `max_deviation` from fit parameters
2. Generates predictions on test set
3. Creates 2D correlation plots for each class
4. Produces comparison histograms
5. Saves trained model

#### `classify_predicted_integralMaxdev(path_to_classifier_model)`

**Purpose**: Tests classification performance using predicted (rather than true) metrics.

---

## 3. Classify

**Purpose**: Orchestrates complete classification pipeline using trained regression and classification models.

### Constructor

```python
def __init__(self, regressor_model=None, classifier_model=None, path_to_data=None, output_path=None):
```

### Workflow

#### `predMetrics(data_df, plot2dcorr=False)`

**Purpose**: Predicts metrics using regression model.

**Process**:

1. Takes fit parameters as input
2. Predicts `integral_R` and `max_deviation`
3. Optionally generates correlation plots by class

#### `classify_metrics(data_df, plotconfmatrix=False)`

**Purpose**: Classifies samples using predicted metrics.

**Process**:

1. Uses predicted metrics as classifier input
2. Generates class predictions
3. Optionally creates confusion matrix

#### `run_classification(info_data_dict, plot2dcorr=False, plotconfmatrix=False)`

**Purpose**: Executes complete pipeline from data loading to classification results.

---

## 4. preClassifier

**Purpose**: Direct waveform classification without intermediate metric prediction steps.

### Constructor

```python
def __init__(self, path_to_train, output_path):
```

### Key Features

- **Direct Input**: Uses raw waveform data (70 data points) as features
- **End-to-End**: Bypasses fit parameter and metric prediction steps
- **High Performance**: Achieves very high classification accuracy

### Methods

#### `read_npy()`

**Purpose**: Loads and preprocesses numpy-format waveform datasets.

#### `split_train_test_valid(data_df)`

**Purpose**: Splits waveform data into training/testing sets.
**Returns**: Dictionary with training and testing arrays for waveforms, labels, and sample IDs.

#### `train(best_params, X_set, y_set, ids_set)`

**Purpose**: Trains XGBoost classifier on waveform data.

#### `plot_wf_with_class(wf_data, output_path, trueclass, predclass)`

**Purpose**: Visualizes waveforms with classification results.

---

## 5. TestPreclassifier

**Purpose**: Tests pre-trained waveform classifier on data from ROOT files (experimental data format).

### Constructor

```python
def __init__(self, path_to_root_file='', hist_prefix='hist_0'):
```

### Key Capabilities

- **ROOT File Integration**: Reads experimental data from ROOT format
- **Real-Time Classification**: Classifies individual channel responses
- **Visualization**: Generates plots of classified waveforms

### Methods

#### `read_ROOT(filename, hist_prefix)`

**Purpose**: Opens and reads ROOT file data.

#### `get_histogram(root_data, histname, Npoints=70)`

**Purpose**: Extracts histogram data for specific channels.

#### `getCHN_resp(chn)`

**Purpose**: Retrieves channel response waveform.

#### `run(path_to_model, chn)`

**Purpose**: Classifies single channel and saves visualization.

---

## Data Flow Architecture

```
Raw Waveforms → Fit Parameters → Metrics → Classification
     ↓              ↓              ↓           ↓
preClassifier   BDT_Regressor  BDT_Classifier  Final Results
     ↓              ↓              ↓           ↓
 Direct Classify  Predict I&D   Classify     Class Labels
```


**Legend**:

- **I&D**: Integral and Deviation metrics
- **Direct Classify**: Waveform → Class (bypass intermediate steps)
- **Traditional Pipeline**: Waveform → Fit Params → Metrics → Class

## Model Performance Targets

- **Classification Accuracy**: ≥ 99.85%
- **Regression Metrics**: RMSE optimization
- **Cross-Validation**: 3-fold CV for hyperparameter tuning
- **Early Stopping**: Prevents overfitting during training

## Hardware Acceleration

- **GPU Support**: CUDA-enabled XGBoost training
- **Tree Method**: Histogram-based algorithm for efficiency
- **Parallel Processing**: Multi-core hyperparameter optimization

## Output Formats

- **Models**: JSON format for XGBoost model serialization
- **Predictions**: CSV files with truth vs prediction comparisons
- **Visualizations**: PNG plots for confusion matrices and correlations
- **Metrics**: Performance statistics and accuracy reports

## Key Design Principles

1. **Modularity**: Each class handles specific ML tasks
2. **Flexibility**: Supports both traditional ML pipeline and direct classification
3. **Evaluation**: Comprehensive visualization and metrics
4. **Production Ready**: Model serialization and loading capabilities
5. **Scalability**: GPU acceleration and efficient data handling
