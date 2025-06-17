# Document written by Claude.ai

# main.py - Machine Learning Pipeline Orchestrator

## Overview

This script serves as the main entry point for a comprehensive machine learning pipeline designed for waveform analysis and classification. It orchestrates the training of multiple XGBoost models and tests them on both simulated and real experimental data from ROOT files.

## Script Structure

The script is organized as a single `if __name__=='__main__':` block that sequentially executes different phases of the machine learning workflow.

## Execution Phases

### Phase 1: Data Preparation (Commented Out)

```python
# train_test split
# root_path = 'data/labelledData/labelledData'
# split_train_test_dataset(path_to_data=root_path, frac_test=0.2)
```

**Purpose**: Splits the complete dataset into training/validation (80%) and testing (20%) sets.
**Status**: Currently disabled, suggesting data has already been split.

---

### Phase 2: Classification Model Training

```python
path_to_data = 'data/labelledData/labelledData_gpuSamples_alot/'
output_path = 'OUTPUT/bdt'

classifier = BDT_Classifier(path_to_data=path_to_data, output_path=output_path)
params = classifier.tune_hyperamaters()
classifier.train(params=params)
```

**Purpose**: Trains an XGBoost classification model.

**Process**:

1. **Data Source**: Uses large GPU-generated sample dataset
2. **Output Directory**: Creates necessary subdirectories in `OUTPUT/bdt/`
3. **Hyperparameter Optimization**: Performs randomized search for optimal parameters
4. **Model Training**: Trains classifier with optimized parameters
5. **Model Evaluation**: Tests accuracy and saves model if performance threshold met

**Target**: Classify samples into 4 classes (c1, c2, c3, c4) based on computed metrics

---

### Phase 3: Regression Model Training

```python
regressor = BDT_Regressor(path_to_data=path_to_data, output_path='OUTPUT/bdt')
params = regressor.tune_hyperparameters()
regressor.train(params=params)
regressor.classify_predicted_integralMaxdev(path_to_classifier_model='OUTPUT/bdt/classifier_bdt_model.json')
```

**Purpose**: Trains an XGBoost regression model and validates the complete pipeline.

**Process**:

1. **Regression Training**: Predicts `integral_R` and `max_deviation` from fit parameters
2. **Hyperparameter Tuning**: Optimizes regression model parameters
3. **Model Training**: Trains regressor with cross-validation
4. **Pipeline Validation**: Tests classification using predicted (not true) metrics
5. **Performance Analysis**: Generates correlation plots and comparison visualizations

**Target**: Predict continuous metrics that serve as input to the classifier

---

### Phase 4: Real Data Classification

```python
combined_classifier = Classify(
    regressor_model='OUTPUT/bdt/regressor_bdt_model.json',
    classifier_model='OUTPUT/bdt/classifier_bdt_model.json',
    path_to_data='data/labelledData',
    output_path='OUTPUT/bdt/TestOnData'
)

combined_classifier.run_classification(
    info_data_dict={
        'key_in_name': 'fit_results',
        'file_ext': '.csv',
        'sep': ','
    },
    plot2dcorr=True, 
    plotconfmatrix=True
)
```

**Purpose**: Applies trained models to real experimental data.

**Process**:

1. **Model Loading**: Loads both trained regression and classification models
2. **Data Processing**: Reads CSV files containing fit results from real data
3. **Complete Pipeline**: Executes fit parameters → metrics → classification
4. **Visualization**: Generates 2D correlation plots and confusion matrices
5. **Results Export**: Saves classified data and performance metrics

**Input**: Real experimental data with fit parameters
**Output**: Class predictions with comprehensive evaluation

---

### Phase 5: Direct Waveform Classification Testing

```python
test_ = TestPreclassifier(
    path_to_root_file='raw_waveforms_run_30413.root', 
    hist_prefix='hist_0'
)

for chn in range(300):
    test_.run(
        path_to_model='OUTPUT/bdt/preclassifier/preclassifier.json', 
        chn=chn
    )
```

**Purpose**: Tests pre-classifier model on raw experimental waveforms from ROOT files.

**Process**:

1. **ROOT File Access**: Opens experimental data file containing raw waveforms
2. **Channel Iteration**: Processes 300 detector channels
3. **Direct Classification**: Classifies waveforms without intermediate steps
4. **Individual Results**: Generates classification and visualization for each channel
5. **Performance Validation**: Tests pre-classifier accuracy on real detector data

**Input**: Raw waveform histograms from experimental detector
**Output**: Direct class predictions for each detector channel

---

## Data Flow Diagram

```
Experimental Data Pipeline:
┌─────────────────┐
│ Raw Waveforms   │ -> Preclassifier Model -> CLasses
│ (ROOT files)    │
└─────────────────┘

Traditional ML Pipeline:
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Fit Parameters  │ →  │ BDT          │ →  │ Predicted       │ →  │ BDT          │
│                 │    │ Regressor    │    │ Metrics         │    │ Classifier   │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                                           │
                                           ┌─────────────────┐            │
                                           │ Class Labels    │ ←──────────┘
                                           │                 │
                                           └─────────────────┘
```

## Configuration Parameters

### Data Paths

- **Training Data**: `'data/labelledData/labelledData_gpuSamples_alot/'`
- **Real Data**: `'data/labelledData'`
- **ROOT File**: `'raw_waveforms_run_30413.root'`
- **Output**: `'OUTPUT/bdt'`

### Model Parameters

- **Classes**: 4 categories (c1, c2, c3, c4)
- **Channels**: 300 detector channels
- **Features**: Fit parameters and computed metrics
- **Algorithms**: XGBoost with GPU acceleration

### Evaluation Metrics

- **Classification Accuracy**: Target ≥ 99.85%
- **Regression RMSE**: Optimized through cross-validation
- **Visualization**: Confusion matrices and correlation plots

## Output Structure

```
OUTPUT/bdt/
├── classifier_bdt_model.json          # Trained classification model
├── regressor_bdt_model.json           # Trained regression model
├── prediction_testdataset/             # Test results and visualizations
│   ├── confusionMatrix_*.png
│   ├── *_2dCorr.png
│   └── comparison_*.png
├── TestOnData/                         # Real data classification results
│   ├── classified_data.csv
│   ├── predicted_integral_max_dev_ondata.csv
│   └── visualization files
└── preclassifier/                      # Direct waveform classification
    ├── preclassifier.json
    └── testPreclassifier_fromROOT/
```

## Key Features

1. **Multi-Model Training**: Regression and classification models
2. **Hyperparameter Optimization**: Automated parameter tuning
3. **Comprehensive Evaluation**: Multiple visualization and metrics
4. **Real Data Testing**: Validation on experimental detector data
5. **Flexible Pipeline**: Both traditional ML and direct classification approaches
6. **Production Ready**: Model serialization and systematic output organization

## Dependencies

- **BDT.py**: Core machine learning classes and functions
- **XGBoost**: Gradient boosting framework
- **uproot**: ROOT file reading capability
- **pandas, numpy**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **scikit-learn**: ML utilities and metrics

## Usage Notes

- The script assumes pre-existing data directories and files
- GPU acceleration is enabled for XGBoost training
- Some sections are commented out, indicating they may be run selectively
- The pipeline is designed for a specific detector physics application
- Results are automatically saved with comprehensive logging and visualization
