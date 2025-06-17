# Document written by Claude.ai

# Sim_waveform Class Documentation

## Overview

The `Sim_waveform` class is designed to generate synthetic waveforms using fit parameters and response functions. This class is primarily used for creating training datasets for machine learning models that work with waveform data.

## Class Definition

```python
class Sim_waveform:
    '''
    Generate waveforms using the fit parameters and the response function.
    '''
```

## Constructor

```python
def __init__(self, path_to_sim=None, output_path=None):
```

### Parameters

- **`path_to_sim`** *(str, optional)*: Path to the CSV file containing simulation parameters
- **`output_path`** *(str, optional)*: Directory path where generated waveforms will be saved

### Attributes

- **`path_to_sim`**: Input CSV file path
- **`output_path`**: Output directory path
- **`sim_data`**: Pandas DataFrame containing simulation data (loaded from CSV if path provided)
- **`response_params`**: List of parameter names used for the response function:
  `['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']`

## Methods

### `read_csv_sim()`

```python
def read_csv_sim(self):
```

**Purpose**: Reads and preprocesses the simulation data from CSV file.

**Returns**: Pandas DataFrame with cleaned data (removes unnamed columns, converts channel numbers to int32)

**Processing**:

- Removes columns with "Unnamed" in their names
- Converts '#Ch.#' column to int32 data type

---

### `__generate_1wf(params=None)` (Private Method)

```python
def __generate_1wf(self, params=None):
```

**Purpose**: Generates a single waveform using provided parameters.

**Parameters**:

- **`params`** *(list)*: List of response function parameters

**Returns**:

- Numpy array representing the response waveform, or `None` if no parameters provided

**Process**:

- Creates a time array from `params[0]` to `params[0]+70` with 70 points
- Calls the `response()` function (imported from external module) to generate waveform

---

### `run()`

```python
def run(self):
```

**Purpose**: Generates individual waveform files (.npz format) for each sample in the dataset.

**Process**:

1. Iterates through each sample in `sim_data`
2. Extracts response parameters for each sample
3. Generates waveform using `__generate_1wf()`
4. Creates a dictionary containing:
   - Response parameters
   - Channel number
   - Class label
   - Generated waveform
   - Integral and max deviation values
5. Saves each sample as separate `.npz` file with naming pattern: `wf_{key_name}_{sample_index}.npz`

---

### `data2npy()`

```python
def data2npy(self):
```

**Purpose**: Generates waveforms and saves the entire enriched dataset to a single `.npy` file.

**Process**:

1. Processes up to 50,000 samples from the dataset
2. For each sample:
   - Generates waveform using response parameters
   - Creates enriched dictionary with all sample data plus generated waveform
3. Saves all enriched data as a single `.npy` file
4. Prints confirmation message with output file path

**Output**: Single `.npy` file containing list of dictionaries, each representing one enriched sample

## Usage Example

```python
# Initialize with simulation data
sim_wf = Sim_waveform(
    path_to_sim='data/simulation_params.csv',
    output_path='output/waveforms/'
)

# Generate individual .npz files for each sample
sim_wf.run()

# Or generate a single .npy file with all data
sim_wf.data2npy()
```

## Key Features

- **Flexible Output**: Supports both individual file output (`.npz`) and batch output (`.npy`)
- **Parameter-Driven**: Uses response function parameters to generate realistic waveforms
- **Data Enrichment**: Combines original simulation parameters with generated waveforms
- **Scalable**: Can handle large datasets (configured for up to 50,000 samples)
- **Class Integration**: Preserves class labels for supervised learning applications

## Dependencies

- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **response module**: External module containing the `response()` function for waveform generation

## Notes

- The class assumes a specific response function that takes time array and parameters
- Generated waveforms have a fixed length of 70 data points
- The class is designed to work with pre-labeled simulation data containing class information
- File naming includes a key extracted from the input CSV filename for organization
