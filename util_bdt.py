import os, sys
import torch
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

    # print(encoded)
    # print(feature_names)
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
    # print(original_df)
    # sys.exit()
    X = original_df[cols_input]
    y = original_df[cols_output]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42
    )

    output = {
        'regressor' : {
            'X_train': X_train,# / np.abs(X_train).max(),
            'y_train': y_train[cols_output_regressor],# / np.abs(y_train[cols_output_regressor]).max(),
            'X_val' : X_val,# / np.abs(X_val).max(),
            'y_val' : y_val[cols_output_regressor],# / np.abs(y_val[cols_output_regressor]).max(),
            'X_test' : X_test,# / np.abs(X_test).max(),
            'y_test' : y_test[cols_output_regressor],# / np.abs(y_test[cols_output_regressor]).max()
        },
        'classifier' : {
            'X_train' : y_train[cols_output_regressor],# / np.abs(y_train[cols_output_regressor]).max(),
            'y_train' : y_train[cols_output_classifier],
            'X_val' : y_val[cols_output_regressor],# / np.abs(y_val[cols_output_regressor]).max(),
            'y_val' : y_val[cols_output_classifier],
            'X_test' : y_test[cols_output_regressor],# / np.abs(y_test[cols_output_regressor]).max(),
            'y_test' : y_test[cols_output_classifier]
        }
    }
    return output

def dataframe2DMatrix(X, y=None, multiOutputClassifier=False):
    if multiOutputClassifier:
        return xgb.DMatrix(X, label=y, enable_categorical=True)
    else:
        return xgb.DMatrix(X, label=y)

from xgboost.callback import TrainingCallback

class LearningRateDecay(TrainingCallback):
    """Custom learning rate decay callback for XGBoost"""
    
    def __init__(self, initial_lr, decay_factor, decay_rounds):
        """
        Parameters:
        -----------
        initial_lr : float
            Initial learning rate
        decay_factor : float
            Factor to multiply learning rate by each decay_rounds
        decay_rounds : int
            Number of rounds between learning rate updates
        """
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_rounds = decay_rounds
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration"""
        if (epoch > 0) and (epoch % self.decay_rounds == 0):
            new_lr = self.initial_lr * (self.decay_factor ** (epoch // self.decay_rounds))
            model.set_param('learning_rate', new_lr)
        return False

def gridSearch_Regressor(train_data_dict, param_grid: dict, item_to_predict: str, regressor=True):
    """
    Performs grid search to find optimal hyperparameters for XGBoost using native API.
    
    Parameters:
    -----------
    train_data_dict : dict
        Dictionary containing training and validation data with keys:
        - 'X_train': Training input features
        - 'y_train': Training target values
        - 'X_val': Validation input features
        - 'y_val': Validation target values
    
    param_grid : dict
        Dictionary with hyperparameter names as lists of values to try.
        Example:
        {
            'max_depth': [3, 4, 5],
            'eta': [0.01, 0.1],      # learning_rate
            'num_boost_round': [100, 200]
        }
    
    Returns:
    --------
    dict
        Dictionary containing the best parameters found during grid search
    """
    # Convert to DMatrix format
    dtrain = None
    dval = None
    if regressor:
        dtrain = dataframe2DMatrix(X=train_data_dict['X_train'], y=train_data_dict['y_train'][item_to_predict])
        dval = dataframe2DMatrix(X=train_data_dict['X_val'], y=train_data_dict['y_val'][item_to_predict])
    else:
        dtrain = dataframe2DMatrix(train_data_dict['X_train'], y=train_data_dict['y_train'])
        dval = dataframe2DMatrix(train_data_dict['X_val'], y=train_data_dict['y_val'])

    objective = 'reg:squarederror'
    eval_metric = 'rmse'
    if not regressor:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    # Base parameters that won't be tuned
    base_params = {
        'objective': objective,
        # 'tree_method': 'gpu_hist',
        'num_boost_round': 200,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': eval_metric
    }
    
    # Initialize tracking of best model
    best_score = float('inf')
    best_params = None
    best_num_round = None
    
    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        # Create parameter dictionary for this iteration
        current_params = dict(zip(param_names, values))
        current_params.update(base_params)
        
        # Extract num_boost_round if present, otherwise use default
        num_boost_round = current_params.pop('num_boost_round', 100)
        
        print(f"\nTrying parameters: {current_params}")
        print(f"Number of rounds: {num_boost_round}")
        
        initial_lr = current_params.get('learning_rate', 0.3)
        
        # Create proper callback instance
        lr_callback = LearningRateDecay(
            initial_lr=initial_lr,
            decay_factor=0.75,  # 5% decay
            decay_rounds=20     # every 50 rounds
        )

        # Train model with current parameters
        evals_result = {}
        model = xgb.train(
            params=current_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=100,
            evals_result=evals_result,
            verbose_eval=False,
            callbacks=[lr_callback]
        )
        
        # Get best validation score
        final_score = None
        if regressor:
            final_score = min(evals_result['eval']['rmse'])
            print(f"Validation RMSE: {final_score:.4f}")
            
            # Update best parameters if better score found
            if final_score < best_score:
                best_score = final_score
                best_params = current_params
                best_num_round = model.best_iteration
                print("New best score!")
        else:
            final_score = min(evals_result['eval']['logloss'])
            print(f"Validation logloss: {final_score:.4f}")
            
            # Update best parameters if better score found
            if final_score < best_score:
                best_score = final_score
                best_params = current_params
                best_num_round = model.best_iteration
                print("New best score!")

    best_params['num_boost_round'] = best_num_round
    print("\nBest parameters found:")
    print(f"Parameters: {best_params}")
    print(f"Best RMSE: {best_score:.4f}")
    
    return best_params

def XGBRegressor_model(best_params):
    xgbReg_model = XGBRegressor(random_state=42,
                                **best_params)
    
    modelRegressor = MultiOutputRegressor(xgbReg_model)
    return modelRegressor

#-----------------------------
## Implementation of the Kernel Density Estimation
## from the math
## Current kernel function available: gaussian
class KernelDensityEstimation:
    '''
        Implementation of the Kernel Density Estimation using pytorch for 1d and 2d cases.
        The chosen kernel here is a gaussian.
        inputs:
            data: a 1d or 2d tensor of the original data,
            kernel_func: kernel function to be used. Available function : gaussian,
            bw: bandwidth of the kernel. bw is a scalar that control the smoothness of the probability density in case of 1d sampling.
                For 2d sampling, bw is the covariance matrix of the original data in order to capture the correlation between the variables.
    '''
    def __init__(self, data=None, kernel_func='gaussian', bw=0.1, dim=1):
        self.kernel_func = kernel_func
        self.data = data
        self.dim = dim
        print(self.dim)
        if self.dim==2:
            data_centered = self.data - self.data.mean(dim=1, keepdim=True)
            # cov_matrix = data_centered @ data_centered.T / (self.data.shape[1] - 1)
            cov_matrix = torch.tensor(np.cov(np.array(self.data)), dtype=torch.float32)
            self.bandwidth = cov_matrix * (self.data.shape[1]**(-1/(self.dim+4)))  # Scott's rule
            # self.bandwidth = torch.tensor(np.cov(np.array(self.data)), dtype=torch.float32)
        else:
            self.bandwidth = bw

    def __kernel_func(self, x, xi, bw):
        if self.kernel_func=='gaussian':
            if self.dim==1:
                c1 = 1/(bw*np.sqrt(2*np.pi))
                c2 = np.exp(-0.5*np.power((x-xi)/(bw), 2))
                return c1*c2 
            elif self.dim==2:
                X = x - xi
                c1 = 1 / (torch.det(bw)*torch.sqrt(torch.tensor(2, dtype=torch.float32)*torch.pi))
                arg = X @ torch.inverse(bw) @ X
                c2 = torch.exp(-torch.tensor(0.5, dtype=torch.float32)*arg).item()
                return c1*c2
            
    def eval_density(self, x):
        out = 0
        if self.dim==1:
            N = len(self.data)
            for i in range(N):
                out += self.__kernel_func(x=x, xi=self.data[i].item(), bw=self.bandwidth)
            out = out / N
            return out
        elif self.dim==2:
            N = self.data.shape[1]
            out = torch.sum(torch.tensor([self.__kernel_func(x=x, xi=self.data[:, i], bw=self.bandwidth) for i in range(N)]))
            out = out / N
            return out.item()
            

    def resample(self, N_samples=None):
        if N_samples is None:
            print("Invalid Number of samples.")
            return None
        if self.dim==1:
            x_grid = np.linspace(torch.min(self.data).item(), torch.max(self.data).item(), N_samples)
            probabilities = self.eval_density(x=x_grid)
            indices = torch.multinomial(torch.tensor(probabilities, dtype=torch.float32), N_samples, replacement=True)
            new_samples = x_grid[indices]
            return x_grid, probabilities, new_samples
        elif self.dim==2:
            x_grid = torch.tensor(np.random.uniform(torch.min(self.data[0]).item(), torch.max(self.data[0]).item(), N_samples), dtype=torch.float32)
            y_grid = torch.tensor(np.random.uniform(torch.min(self.data[1]).item(), torch.max(self.data[1]).item(), N_samples), dtype=torch.float32)
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            probabilities = [self.eval_density(x=xy) for xy in xy_grid]
            indices = torch.multinomial(torch.tensor(probabilities, dtype=torch.float32), N_samples, replacement=True)
            selected_samples = xy_grid[indices].T
            return xy_grid, probabilities, selected_samples