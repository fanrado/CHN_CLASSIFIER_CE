import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix

import seaborn as sns

def one_hot_encode_sklearn(data, column_name):
    """
    One-hot encode using scikit-learn's OneHotEncoder.
    
    Parameters:
    data (pandas.DataFrame): Input DataFrame
    column_name (str): Name of the column to encode
    
    Returns:
    pandas.DataFrame: DataFrame with one-hot encoded column
    """
    # Reshape data for encoder
    X = data[column_name].values.reshape(-1, 1)
    
    # Create and fit the encoder
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(X)
    
    # Create new column names
    feature_names = encoder.get_feature_names_out([column_name])

    print(encoded)
    print(feature_names)
    # Create DataFrame with encoded values
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
    
    # Combine with original data
    result = pd.concat([data.drop(column_name, axis=1), encoded_df], axis=1)
    
    return result

def train_valid_test(original_df=None, cols_input=None, cols_output=None, cols_output_regressor=None, cols_output_classifier=None):
    """
    Splits the dataset into training, validation, and test sets for both regression and classification tasks.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        The original dataframe containing all input and output features
    cols_input : list
        List of column names to be used as input features
    cols_output : list
        List of all output column names (both regression and classification targets)
    cols_output_regressor : list
        List of column names for regression targets
    cols_output_classifier : list
        List of column names for classification targets
    
    Returns:
    --------
    dict
        A dictionary containing two sub-dictionaries for 'regressor' and 'classifier', each with:
        - X_train: Training input features
        - y_train: Training target values
        - X_val: Validation input features
        - y_val: Validation target values
        - X_test: Test input features
        - y_test: Test target values
        
    Note:
    -----
    - Uses a 80-20 split for test set
    - Further splits the remaining 80% into 80-20 for train-validation
    - Final split ratio is approximately 64-16-20 (train-validation-test)
    - For classifier, uses regression outputs as input features
    """
    if (original_df is None) or (cols_input is None) or (cols_output is None) or (cols_output_regressor is None) or (cols_output_classifier is None):
        return None
    
    X = original_df[cols_input]
    y = original_df[cols_output]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42
    )

    output = {
        'regressor' : {
            'X_train': X_train / X_train.max(),
            'y_train': y_train[cols_output_regressor] / y_train[cols_output_regressor].max(),
            'X_val' : X_val / X_val.max(),
            'y_val' : y_val[cols_output_regressor] / y_val[cols_output_regressor].max(),
            'X_test' : X_test / X_test.max(),
            'y_test' : y_test[cols_output_regressor] / y_test[cols_output_regressor].max()
        },
        'classifier' : {
            'X_train' : y_train[cols_output_regressor],
            'y_train' : y_train[cols_output_classifier],
            'X_val' : y_val[cols_output_regressor],
            'y_val' : y_val[cols_output_classifier],
            'X_test' : y_test[cols_output_regressor],
            'y_test' : y_test[cols_output_classifier]
        }
    }
    return output

def dataframe2DMatrix(X, y=None):
    return xgb.DMatrix(X, label=y)

def gridSearch_Regressor(train_data_dict, param_grid: dict):
    """
    Performs grid search cross-validation to find optimal hyperparameters for XGBoost regressor.
    
    Parameters:
    -----------
    train_data_dict : dict
        Dictionary containing training and validation data with keys:
        - 'X_train': Training input features
        - 'y_train': Training target values
        - 'X_val': Validation input features
        - 'y_val': Validation target values
    
    param_grid : dict
        Dictionary with hyperparameter names as keys and lists of parameter values to try.
        Example:
        {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200]
        }
    
    Returns:
    --------
    dict
        Dictionary containing the best parameters found during grid search
    
    Notes:
    ------
    - Uses GPU acceleration with 'gpu_hist' tree method
    - Performs 3-fold cross-validation
    - Uses negative mean squared error as scoring metric
    - Utilizes all available CPU cores (n_jobs=-1)
    - Prints best parameters and score during execution
    
    Example:
    --------
    >>> param_grid = {
    ...     'max_depth': [3, 4, 5],
    ...     'learning_rate': [0.01, 0.1]
    ... }
    >>> best_params = gridSearch_Regressor(train_data, param_grid)
    """
    # Initialize the XGBoost regressor
    xgb_reg = XGBRegressor(
        tree_method='gpu_hist',
        enable_categorical=True,
    )

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # or 'r2', 'neg_mean_absolute_error'
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(
        train_data_dict['X_train'],
        train_data_dict['y_train'],
        eval_set=[(train_data_dict['X_val'], train_data_dict['y_val'])],
        verbose=True
    )

    # Get the best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", -grid_search.best_score_)  # Convert back to positive MSE

    return grid_search.best_params_

def XGBRegressor_model(best_params):
    xgbReg_model = XGBRegressor(random_state=42,
                                **best_params)
    
    modelRegressor = MultiOutputRegressor(xgbReg_model)
    return modelRegressor




