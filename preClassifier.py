# Import libraries
import os, sys
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, uniform

from response import *

import xgboost as xgb
from util_bdt import dataframe2DMatrix, one_hot_encode_sklearn

from sklearn.metrics import confusion_matrix

import seaborn as sns
import random

# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.model_selection import RandomizedSearchCV

class Sim_waveform:
    '''
        Generate waveforms using the fit parameters and the response function.
    '''
    def __init__(self, path_to_sim=None, output_path=None):
        self.path_to_sim = path_to_sim
        self.output_path = output_path
        self.sim_data = pd.DataFrame()
        if self.path_to_sim is not None:
            self.sim_data = self.read_csv_sim()
        # self.response_params = ['#Ch.#', 't', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.response_params = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']

    def read_csv_sim(self):
        tmpdata = pd.read_csv(self.path_to_sim)
        cols = list(tmpdata.columns)
        cols_to_be_removed = [c for c in cols if 'Unnamed' in c]
        tmpdata.drop(labels=cols_to_be_removed, axis=1, inplace=True)
        tmpdata['#Ch.#'] = tmpdata['#Ch.#'].astype('int32')
        return tmpdata
    
    def __generate_1wf(self, params=None):
        if params is None:
            return None
        x = np.linspace(params[0], params[0]+70, 70)
        R = response(x=x, par=params)
        return R
    
    def run(self):
        N_samples = self.sim_data.shape[0]
        # N_samples = 10000
        key_name = self.path_to_sim.split('/')[-1].split('.')[0].split('_')[3]
        for isample in range(N_samples):
            params = list(self.sim_data[self.response_params].iloc[isample])
            R = self.__generate_1wf(params=params)
            dict_params = dict(zip(self.response_params, params))
            dict_params['#Ch.#'] = self.sim_data['#Ch.#'].iloc[isample]
            dict_params['class'] = self.sim_data['class'].iloc[isample]
            dict_params['wf'] = R
            dict_params['integral_R'] = self.sim_data['integral_R'].iloc[isample]
            dict_params['max_deviation'] = self.sim_data['max_deviation'].iloc[isample]
            np.savez(f'{self.output_path}/wf_{key_name}_{isample}.npz', **dict_params)

    def data2npy(self):
        """
            Save the self.sim_data along with the waveform datapoints to a .npy file.
        """
        enriched_data = []
        N_samples = self.sim_data.shape[0]
        N_samples = 50000
        for isample in range(N_samples):
            params = list(self.sim_data[self.response_params].iloc[isample])
            R = self.__generate_1wf(params=params)
            dict_params = dict(zip(self.response_params, params))
            dict_params['#Ch.#'] = self.sim_data['#Ch.#'].iloc[isample]
            dict_params['class'] = self.sim_data['class'].iloc[isample]
            dict_params['wf'] = R
            dict_params['integral_R'] = self.sim_data['integral_R'].iloc[isample]
            dict_params['max_deviation'] = self.sim_data['max_deviation'].iloc[isample]
            enriched_data.append(dict_params)
        # save the enriched_data to a .npy file
        output_file = self.path_to_sim.split('/')[-1].split('.')[0]
        output_file = f'{self.output_path}/{output_file}.npy'
        np.save(output_file, enriched_data)
        print(f'Data saved to {output_file}')

class Load_chunk_dset:
    '''
        This class will load a chunk of the data.
    '''
    def __init__(self, path_to_dset: str, chunk_size=5, target_columns=[''], input_columns=[], Ntest=5000):
        '''
            - path_to_dset: path to where the .npz files are located.
            - chunk_size: the size of the chunk you want to use.
            - target_columns: what are your target columns ?
            - input_columns: what input columns do you want to use ? If none is provided, the default input columns are the data points in a waveforms (70 points => 70 columns).
            - Ntest: the number of samples per class you want to use for testing.
        '''
        self.path_to_dset = path_to_dset
        # self.list_dset = ['/'.join([self.path_to_dset, f]) for f in os.listdir(self.path_to_dset)[:-Ntest]]
        self.list_dset = ['/'.join([self.path_to_dset, f]) for f in os.listdir(self.path_to_dset)]
        #
        # this split is crucial because it make sure that the list_test has all classes
        self.list_train, self.list_test = self.train_test_split(list_dset=self.list_dset, Ntest=Ntest)

        self.target_columns = target_columns
        if len(input_columns)==0:
            self.input_columns = [f'p{i}' for i in range(70)]
        else:
            self.input_columns = input_columns
        self.input_columns.append('#Ch.#')
        self.chunk_size = chunk_size
        self.iter = 0

    def train_test_split(self, list_dset, Ntest):
        '''
            Split the entire dataset into train and test sets.
            Ntest is the number of samples in each class.
        '''
        c1 = [f for f in list_dset if 'c1' in f]
        c2 = [f for f in list_dset if 'c2' in f]
        c3 = [f for f in list_dset if 'c3' in f]
        c4 = [f for f in list_dset if 'c4' in f]
        # Print the length of each class list
        print(f"Length of class c1: {len(c1)}")
        print(f"Length of class c2: {len(c2)}")
        print(f"Length of class c3: {len(c3)}")
        print(f"Length of class c4: {len(c4)}")
        train_c1, test_c1 = c1[:-Ntest], c1[-Ntest:]
        train_c2, test_c2 = c2[:-Ntest], c2[-Ntest:]
        train_c3, test_c3 = c3[:-Ntest], c3[-Ntest:]
        train_c4, test_c4 = c4[:-Ntest], c4[-Ntest:]
        train = train_c1 + train_c2 + train_c3 + train_c4
        test = test_c1 + test_c2 + test_c3 + test_c4
        random.shuffle(train)
        random.shuffle(test)
        return train, test
    
    def npz2df(self, filepath='', forTest_regressor=False, forTest_classifier=False):
        '''
            This function reads a .npz file and convert it to  dataframe with one row.
        '''
        data = np.load(filepath)
        wf = np.array(data['wf']).reshape(-1, 1)
        wf_dict = {f'p{i}': wf[i] for i in range(len(wf))}
        keys = [c for c in list(data.keys()) if c!='wf']
        for k in keys:
            wf_dict[k] = np.array(data[k].flatten())
        wf_df = pd.DataFrame(wf_dict)

        if forTest_regressor:
            # split features (input) and target values
            y = wf_df[self.target_columns]
            X = wf_df[self.input_columns]
            return X, y
        elif forTest_classifier:
            return wf_df[self.input_columns + self.target_columns]
        else:
            return wf_df

    def reset(self):
        '''
            Reset the iteration index.
        '''
        self.iter = 0
        return self
    
    def load(self, tasktype='regression', forRandomSearchCV=False):
        '''
            This function tries to load a chunk of the dataset. If successful, it returns a dMatrix. Otherwise, it returns None.
        '''
        try:
            chunk = pd.DataFrame()
            if self.chunk_size*(self.iter+1) > len(self.list_train)-1:
                return None
            for i, ichunk in enumerate(range(self.iter*self.chunk_size, (self.iter+1)*self.chunk_size)):
                df = self.npz2df(filepath=self.list_train[ichunk])
                if i==0:
                    chunk = df.copy()
                else:
                    chunk = pd.concat([chunk, df], axis=0)
            if len(chunk)==0:
                return None
            self.iter += 1
            if tasktype=='regression':
                # split features (input) and target values
                # use dataframes
                y = chunk[self.target_columns]
                X = chunk[self.input_columns]
                if forRandomSearchCV:
                    return X, y
                return dataframe2DMatrix(X=X, y=y)
            elif tasktype=='classification':
                tmp_chunk = one_hot_encode_sklearn(data=chunk, column_name='class')
                X = tmp_chunk[self.input_columns]
                y = tmp_chunk[self.target_columns]
                return dataframe2DMatrix(X=X, y=y)
        except:
            return None
    

def randomSearchCV(train_data_dict, param_distributions, n_iter=50, scoring='neg_mean_squared_error', cv=2):
    """
    Manually implement RandomizedSearchCV for xgb.Booster.

    Parameters:
    -----------
    train_data_dict : list
        List containing training input features and target values:
        - train_data_dict[0]: Training input features (X_train)
        - train_data_dict[1]: Training target values (y_train)

    param_distributions : dict
        Dictionary with hyperparameter names as distributions or lists of values to sample from.
        Example:
        {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'num_boost_round': randint(50, 300)
        }

    n_iter : int
        Number of parameter settings to sample.

    scoring : str
        Scoring metric to evaluate the model. Default is 'neg_mean_squared_error'.

    cv : int
        Number of cross-validation folds.

    Returns:
    --------
    dict
        Dictionary containing the best parameters and the corresponding score.
    """
    X_train = train_data_dict[0]
    y_train = train_data_dict[1]

    # Convert data to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Initialize variables to track the best parameters and score
    best_params = None
    best_score = float('inf') if scoring == 'neg_mean_squared_error' else -float('inf')

    # Perform random search
    for i in range(n_iter):
        # Sample a random combination of hyperparameters
        sampled_params = {key: dist.rvs() if hasattr(dist, 'rvs') else random.choice(dist)
                          for key, dist in param_distributions.items()}

        # Extract num_boost_round if present
        num_boost_round = sampled_params.pop('num_boost_round', 100)

        print(f"Iteration {i+1}/{n_iter}: Testing parameters: {sampled_params}")

        # Perform cross-validation
        cv_results = xgb.cv(
            params=sampled_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            nfold=cv,
            metrics=('rmse' if scoring == 'neg_mean_squared_error' else scoring),
            early_stopping_rounds=10,
            seed=42,
            verbose_eval=False
        )

        # Get the mean score from cross-validation
        mean_score = cv_results['test-rmse-mean'].min() if scoring == 'neg_mean_squared_error' else cv_results['test-logloss-mean'].min()

        print(f"Score for iteration {i+1}: {mean_score}")

        # Update the best parameters and score if the current score is better
        if (scoring == 'neg_mean_squared_error' and mean_score < best_score) or \
           (scoring != 'neg_mean_squared_error' and mean_score > best_score):
            best_score = mean_score
            best_params = sampled_params
            best_params['num_boost_round'] = num_boost_round

    print("\nBest parameters found:")
    print(f"Parameters: {best_params}")
    print(f"Best score: {best_score}")

    return {'best_params': best_params, 'best_score': best_score}

class PreClassifier_BDT:
    '''
        Given a waveform as input, this class will:
            - predict the fit parameters
            - classify them into four classes c1, c2, c3, c4
        using a boosted decision tree.
        Inputs:
            path_to_data: path to the dataset used for training the regression and classification models.
                        This dataset located at this path should be a numpy dataset (npz) which contains the wf datapoints,
                        the fit parameters ('t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6'), and the corresponding class ('class').
            output_path : path to where the output will be saved.
    '''
    def __init__(self, path_to_data=None, output_path=None, target_columns=[''], Ntest=10):
        '''
            path_to_data : path to the list of npz data,
            output_path : path to where you want to save the output of the code.
        '''
        self.path_to_data = path_to_data
        self.output_path = output_path
        self.target_columns = target_columns
        self.chunk_size = 10000
        self.list_dset = []
        self.list_test = []
        self.Ntest = Ntest

    def tune_hyperparam_bdt(self):
        params = {
            
        }


    # def regression(self):
    def Train_bdt(self, tasktype='regression'):
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        if tasktype=='classification':
            objective = 'binary:logistic'
            eval_metric = 'logloss'
        params = {
            'objective': objective,
            'eval_metric' : eval_metric,
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 20,
            'learning_rate': 0.4,
            'min_child_weight' : 20,
            'num_boost_round': 200,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }

        data_iter = Load_chunk_dset(path_to_dset=self.path_to_data, chunk_size=self.chunk_size, target_columns=self.target_columns, Ntest=self.Ntest)
        self.list_dset = data_iter.list_train
        # print(self.list_dset)
        self.list_test = data_iter.list_test

        # Random search of the hyperparameters
        # data_iter.chunk_size = int(len(self.list_dset)/10)-1
        X, y = data_iter.load(tasktype='regression', forRandomSearchCV=True)
        # Define the parameter distributions
        param_distributions = {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.99),
            'num_boost_round': randint(50, 300),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }

        # Call the function
        best_result = randomSearchCV(
            train_data_dict=[X, y],  # Replace with your training data
            param_distributions=param_distributions,
            n_iter=4,
            scoring='neg_mean_squared_error',
            cv=2
        )

        print("Best parameters:", best_result['best_params'])
        print("Best score:", best_result['best_score'])
        ##
        # sys.exit()

        # data_iter.reset()
        # data_iter.chunk_size = self.chunk_size
        params = best_result['best_params'] # update parameters

        next_chunk = data_iter.load(tasktype=tasktype)
        eval_chunk = data_iter.load(tasktype=tasktype)
        ## Replace by the custom wrapper
        bdt_model = xgb.train(params=params,
                                    dtrain = next_chunk,
                                    evals=[(next_chunk, 'train'), (eval_chunk, 'eval')],
                                    early_stopping_rounds=20,
                                    xgb_model=None,
                                    verbose_eval=True)
        ##
        next_chunk = data_iter.load(tasktype=tasktype)
        eval_chunk = data_iter.load(tasktype=tasktype)
        while (next_chunk is not None) and (eval_chunk is not None):
            bdt_model = xgb.train(params=params,
                                        dtrain = next_chunk,
                                        xgb_model = bdt_model,
                                        evals=[(next_chunk, 'train'), (eval_chunk, 'eval')],
                                        early_stopping_rounds=20,
                                        verbose_eval=True)
            next_chunk = data_iter.load(tasktype=tasktype)
            eval_chunk = data_iter.load(tasktype=tasktype)
        bdt_model.save_model(f'{self.output_path}/pred_fitParams_model.json')
        # bdt_model.save_model(f'{self.output_path}/pred_fitParams_model.json')

        return bdt_model
        # return bdt_model.model
    
    def testRegressor(self, regressor_predFitParams=None, regressor_predIntegral=None, regressor_predMaxdev=None):
        if regressor_predFitParams is None:
            return None
        
        if regressor_predIntegral is None:
            return None
        
        if regressor_predMaxdev is None:
            return None
        
        print('Running the test predicting the integral and max deviation of the tails. Please wait....')
        # list_dset = ['/'.join([self.path_to_data, f]) for f in os.listdir(self.path_to_data)[-Ntest:-1]]
        # list_dset = self.list_dset[-Ntest:-1]
        list_dset = self.list_test
        target_columns = self.target_columns + ['integral_R', 'max_deviation']
        data_iter = Load_chunk_dset(path_to_dset=self.path_to_data, chunk_size=self.chunk_size, target_columns=target_columns) # chunk_size is not used here because we will call npz2df instead of load
        
        # load the regression model trained to predict the value of the integral
        regressor_predIntegral_model = xgb.Booster()
        regressor_predIntegral_model.load_model(regressor_predIntegral)

        # load the regression model trained to predict the value of the maximum deviation between tails
        regressor_predMaxdev_model = xgb.Booster()
        regressor_predMaxdev_model.load_model(regressor_predMaxdev)

        comparison_df = pd.DataFrame()
        for j, f in enumerate(list_dset):
            # print(f'File number {j}')
            Xtest, ytest = data_iter.npz2df(filepath=f, forTest_regressor=True)
            dtest = dataframe2DMatrix(X=Xtest[data_iter.input_columns])
            # predict the fit parameters using the waveform as input
            predictions = regressor_predFitParams.predict(dtest)
            pred_df = pd.DataFrame({f'{self.target_columns[i]}': predictions.reshape(-1,1)[i] for i in range(len(self.target_columns))})
            
            # predict the integral of the tail using the fit parameters as input
            ytest_fitparams = ytest[['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']]
            ytest_dmatrix = xgb.DMatrix(ytest_fitparams, label=None)
            #
            # prediction of integral using truth information
            pred_integral_ofTruth = regressor_predIntegral_model.predict(ytest_dmatrix)
            # prediction of integral using predicted fit parameters
            ypred_dmatrix = xgb.DMatrix(pred_df[['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']], label=None)
            pred_integral_ofPred = regressor_predIntegral_model.predict(ypred_dmatrix)
            #
            # prediction of the max deviation using the true information
            pred_maxdev_ofTruth = regressor_predMaxdev_model.predict(ytest_dmatrix)
            # prediction of the max deviation using the predicted fit parameters
            pred_maxdev_ofPred = regressor_predMaxdev_model.predict(ypred_dmatrix)

            # print('\n')
            # print(pred_integral[0], ytest['integral_R'].iloc[0])

            cols = [f'{c}_pred' for c in pred_df.columns]
            pred_df.columns = cols
            cols = [f'{c}_truth' for c in ytest_fitparams.columns]
            ytest_fitparams.columns = cols

            # concatenate
            pred_df = pd.concat([ytest_fitparams, pred_df], axis=1)
            # print(Xtest.columns)
            # print(ytest.columns)
            pred_df['#Ch.#'] = [Xtest['#Ch.#'].iloc[0]]
            # integral
            pred_df['integral_R_truth_truth'] = [ytest['integral_R'].iloc[0]]
            pred_df['integral_R_truth_pred'] = pred_integral_ofTruth
            pred_df['integral_R_pred_pred'] = pred_integral_ofPred
            # max deviation
            pred_df['max_deviation_truth_truth'] = [ytest['max_deviation'].iloc[0]]
            pred_df['max_deviation_truth_pred'] = pred_maxdev_ofTruth
            pred_df['max_deviation_pred_pred'] = pred_maxdev_ofPred
            
            if j==0:
                comparison_df = pred_df.copy()
            else:
                comparison_df = pd.concat([comparison_df, pred_df.copy()], axis=0)
        # save the output file to a csv
        comparison_df.to_csv(f'{self.output_path}/predictions_fitparams_integral_maxdev.csv', index=True)
        return comparison_df
        
    def testClassification(self, classifier_model_path=None, pred_int_maxDev_df=None): # a full chain test might be easier to implement : regression + classification
        '''
            Use the classifier model trained during classification to classify the interal and maximum deviation predicted by the regression model trained during pre-classification: true waveform -> predicted fit parameters -> predicted integral and max deviation -> class.
        '''
        print('Running test classifier....')
        # list_dset = self.list_dset[-Ntest:-1]
        list_dset = self.list_test
        target_columns = ['class']
        input_columns = self.target_columns + ['integral_R', 'max_deviation', '#Ch.#']
        # input_columns = [f'{c}_truth' for c in tmp_input_columns]
        data_iter = Load_chunk_dset(path_to_dset=self.path_to_data, chunk_size=0, target_columns=target_columns, input_columns=input_columns, Ntest=self.Ntest)
        #
        # load classifier model
        classifier_model = xgb.Booster()
        classifier_model.load_model(classifier_model_path)

        # this is the truth. We can concatenate the rows because there's no waveform data points involved here.
        truth_df = pd.DataFrame()
        for ifile, file in enumerate(list_dset):
            Xy_df = data_iter.npz2df(filepath=file, forTest_classifier=True, forTest_regressor=False)
            if ifile==0:
                truth_df = Xy_df.copy()
            else:
                truth_df = pd.concat([truth_df, Xy_df], axis=0)
        truth_df = one_hot_encode_sklearn(data=truth_df, column_name='class')
        # target_columns = [f'class_c{i}' for i in range(1, 5)]
        #
        # the predicted integral and max deviation along with the channel number is saved in pred_int_maxDev_df (argument).
        output_regressor_pred_df = pred_int_maxDev_df[['integral_R_pred_pred', 'max_deviation_pred_pred']]
        output_regressor_pred_df.columns = ['integral_R', 'max_deviation']
        d_output_regressor_pred = dataframe2DMatrix(X=output_regressor_pred_df)
        pred_classes = classifier_model.predict(d_output_regressor_pred)
        #
        # create a dataframe of the predicted classes
        columns = [cl for cl in truth_df.columns if 'class' in cl]
        predClass_df = pd.DataFrame(pred_classes, index=output_regressor_pred_df.index, columns=columns)
        predClass_df['predicted_class'] = predClass_df.idxmax(axis=1)
        #
        # concatenate channel numbers to the created dataframe
        predClass_df = pd.concat([predClass_df, pred_int_maxDev_df['#Ch.#']], axis=1, join='inner')
        #
        print(predClass_df)
        truth_df['trueClass'] = pd.DataFrame(truth_df[columns].idxmax(axis=1), columns=['trueClass'])
        # print(truth_df)
        # combine truth and predicted class dataframe
        # combined_df = pd.concat([truth_df['#Ch.#'], pd.DataFrame(truth_df[columns].idxmax(axis=1), columns=['trueClass']), predClass_df], axis=1, join='inner')
        # combined_df = pd.concat([truth_df[['#Ch.#', 'trueClass']], predClass_df[['#Ch.#', 'predicted_class']]], axis=1, join='outer')
        #
        # try to concatenate all the truth information with the whole predicted data
        combined_df = pd.concat([truth_df, predClass_df], axis=1, join='outer')
        combined_df.to_csv(f'{self.output_path}/truth_and_prediction_fromClassifier.csv', index=True)
        #
        # Create confusion matrix
        cm = confusion_matrix(y_true=combined_df['trueClass'], y_pred=combined_df['predicted_class'])

        # Create a figure and axis
        plt.figure(figsize=(10, 8))

        # Create heatmap
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='crest',
                    xticklabels=columns,
                    yticklabels=columns)

        # Add labels
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Show plot
        plt.tight_layout()
        plt.savefig('OUTPUT/Preclassifier/confusionMatrix_.png')
        plt.close()


def compare_truth_pred(test_df: pd.DataFrame, output_path: str):
    param = 'integral_R'
    plt.figure()
    plt.hist(test_df[f'{param}_truth_pred'], histtype='step', bins=100, label='truth pred')
    plt.hist(test_df[f'{param}_pred_pred'], histtype='step', bins=100, label='pred pred')
    plt.hist(test_df[f'{param}_truth_truth'], histtype='step', bins=100, label='truth truth')
    plt.title(param)
    # plt.xlim([-2000, 2000])
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(f'{output_path}/comparison_integral_R.png')
    plt.close()
    #
    param = 'max_deviation'
    plt.figure()
    plt.hist(test_df[f'{param}_truth_pred'], histtype='step', bins=100, label='truth pred')
    plt.hist(test_df[f'{param}_pred_pred'], histtype='step', bins=100, label='pred pred')
    plt.hist(test_df[f'{param}_truth_truth'], histtype='step', bins=100, label='truth truth')
    plt.title(param)
    # plt.xlim([-250, 250])
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.savefig(f'{output_path}/comparison_max_deviation.png')
    plt.close()
    

import uproot
class TestPreclassifier:
    """
        This class will be used for the following tasks:
            - open a root file.
            - Get one 1d histogram of the channel response and store it in a numpy array.
            - Load the xgb model to predict the fit parameters -> Store them in a dictionary.
            - Load the xgb models to predict the integral of the tail and the maximum deviation of the tail from the ideal response using the predicted fit parameters -> Save them in a dictionary.
            - Load the xgb model to predict the class -> Save it in a dictionary.
            --> Save all of the predicted items in a dictionary named "predictions"
    """
    def __init__(self, path_to_root_file='', hist_prefix='hist_0'):
        self.path_to_root_file = path_to_root_file
        self.hist_prefix = hist_prefix
        self.root_data = self.read_ROOT(filename=self.path_to_root_file, hist_prefix=hist_prefix)
        self.chn_response = None
        self.predictions = None
        self.__fit_params = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
    
    def read_ROOT(self, filename, hist_prefix='hist_0'):
        root_file = uproot.open(filename)
        return root_file
    
    def get_histogram(self, root_data, histname):
        TH1D_hist = root_data[histname].to_numpy()
        return TH1D_hist
    
    def getCHN_resp(self, chn):
        histname = '_'.join([self.hist_prefix, 'channel', f'{chn};1'])
        hist = self.get_histogram(root_data=self.root_data, histname=histname)
        wf_dict = {f'p{i}': [hist[0][i]] for i in range(70)}
        wf_dict['#Ch.#'] = [chn]
        wf_df = pd.DataFrame(wf_dict)
        return wf_df

    def predict_fitParams(self, path_pred_fitParams_model,   chn_resp_hist):
        pred_fitParams_model = xgb.Booster()
        pred_fitParams_model.load_model(path_pred_fitParams_model)
        print('here')
        dtest = dataframe2DMatrix(X=chn_resp_hist)
        # predict the fit parameters using the waveform as input
        predicted_fitParams = pred_fitParams_model.predict(dtest)
        predicted_fitParams_df = pd.DataFrame({f'{key}': predicted_fitParams.reshape(-1, 1)[i] for i, key in enumerate(self.__fit_params)})
        # print(predicted_fitParams_df)
        # return predicted_fitParams.flatten()
        return predicted_fitParams_df

    def predict_integral_tailR(self, path_pred_integralR_model, fit_params_df):
        pred_integralR_model = xgb.Booster()
        pred_integralR_model.load_model(path_pred_integralR_model)
        dtest = dataframe2DMatrix(X=fit_params_df)
        predicted_integralR = pred_integralR_model.predict(dtest)
        return predicted_integralR.flatten()

    def predict_max_dev_tail(self, path_pred_max_deviation_model, fit_params_df):
        pred_max_deviation_model = xgb.Booster()
        pred_max_deviation_model.load_model(path_pred_max_deviation_model)
        dtest = dataframe2DMatrix(X=fit_params_df)
        predicted_max_deviation = pred_max_deviation_model.predict(dtest)
        return predicted_max_deviation.flatten()

    def predict_class(self, path_classifier_int_maxdev_model, integral_R, max_deviation):
        classifier_int_maxdev_model = xgb.Booster()
        classifier_int_maxdev_model.load_model(path_classifier_int_maxdev_model)
        # output_regressor_pred_df.columns = ['integral_R', 'max_deviation']
        test_df = pd.DataFrame({'integral_R': integral_R, 'max_deviation': max_deviation})
        d_output_regressor_pred = dataframe2DMatrix(X=test_df)
        pred_classes = classifier_int_maxdev_model.predict(d_output_regressor_pred)
        # print(pred_classes)
        # columns = [cl for cl in truth_df.columns if 'class' in cl]
        columns = [f'class_c{i}' for i in range(1, 5)]
        predClass_df = pd.DataFrame(pred_classes, index=test_df.index, columns=columns)
        predClass_df['predicted_class'] = predClass_df.idxmax(axis=1)
        predClass_df.drop(columns, axis=1, inplace=True)
        output_df = pd.concat([test_df, predClass_df], axis=1)
        output_df['predicted_class'] = output_df['predicted_class'].apply(lambda x: x.split('_')[1])
        print(output_df)
        return output_df

if __name__=='__main__':
    # Generating training dataset
    path_to_simdata = 'data/labelledData/labelledData/'
    path_to_simdata = 'data/labelledData/labelledData_gpuSamples_alot'
    # class c1
    print('Generating wf for class c1...')
    filename = 'generate_new_samples_c1.csv'
    # sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generatedSamples/generated_new_samples_c1_labelled_tails.csv',
    #                       output_path='data/labelledData/labelledData/WF_sim/')
    sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                          output_path=f'{path_to_simdata}/npy/')
    sim_wf_obj.data2npy()

    # class c2
    print('Generating wf for class c2...')
    filename = 'generate_new_samples_c2.csv'
    sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                          output_path=f'{path_to_simdata}/npy/')
    sim_wf_obj.data2npy()

    # class c3
    print('Generating wf for class c3...')
    filename = 'generate_new_samples_c3.csv'
    sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                          output_path=f'{path_to_simdata}/npy/')
    sim_wf_obj.data2npy()

    # class c4
    print('Generating wf for class c4...')
    filename = 'generate_new_samples_c4.csv'
    sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                          output_path=f'{path_to_simdata}/npy/')
    sim_wf_obj.data2npy()

    # # class c2
    # print('Generating wf for class c2...')
    # sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generatedSamples/generated_new_samples_c2_labelled_tails.csv',
    #                       output_path='data/labelledData/labelledData/WF_sim/')
    # sim_wf_obj.run()

    # # class c3
    # print('Generating wf for class c3...')
    # sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generatedSamples/generated_new_samples_c3_labelled_tails.csv',
    #                       output_path='data/labelledData/labelledData/WF_sim/')
    # sim_wf_obj.run()

    # # class c4
    # print('Generating wf for class c4...')
    # sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generatedSamples/generated_new_samples_c4_labelled_tails.csv',
    #                       output_path='data/labelledData/labelledData/WF_sim/')
    # sim_wf_obj.run()

    # # Training the regression models to predict the maximum deviation and integral of the tails
    # target_columns = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
    # # target_columns = ['class_c3']
    # # chunk_dset_obj = Load_chunk_dset(path_to_dset='data/labelledData/labelledData/WF_sim', chunk_size=5, target_columns=taget_columns)
    # # chunk_dset_obj.test()
    # preclassifier_obj = PreClassifier_BDT(path_to_data='data/labelledData/labelledData_gpuSamples_alot/WF_sim', output_path='OUTPUT/Preclassifier', target_columns=target_columns, Ntest=10000)
    # regressor_model = preclassifier_obj.Train_bdt(tasktype='regression')
    # #
    # # Test the regression model and compare the result with the truth
    # test_df = preclassifier_obj.testRegressor(regressor_predFitParams=regressor_model, regressor_predIntegral='OUTPUT/synthetic/integral_R_model.json',
    #                                         regressor_predMaxdev='OUTPUT/synthetic/max_deviation_model.json')
    # preclassifier_obj.testClassification(classifier_model_path='OUTPUT/synthetic/classifier_resp_model.json', pred_int_maxDev_df=test_df)
    # # compare truth with prediction
    # compare_truth_pred(test_df=test_df, output_path='OUTPUT/Preclassifier')

    # ## TEST PRE-CLASSIFIER
    # test_ = TestPreclassifier(path_to_root_file='raw_waveforms_run_30413.root', hist_prefix='hist_0')
    # hist_test = test_.getCHN_resp(chn=0)
    # h = hist_test[[f'p{i}' for i in range(70)]].iloc[0]
    # pred_par = test_.predict_fitParams(path_pred_fitParams_model='OUTPUT/Preclassifier/pred_fitParams_model.json', chn_resp_hist=hist_test)
    # pred_integralR = test_.predict_integral_tailR(path_pred_integralR_model='OUTPUT/Kept_RESULTS/Classification_result_may26_GOOD/integral_R_model.json',
    #                              fit_params_df=pred_par[['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']])
    # pred_max_dev = test_.predict_max_dev_tail(path_pred_max_deviation_model='OUTPUT/Kept_RESULTS/Classification_result_may26_GOOD/max_deviation_model.json',
    #                                           fit_params_df=pred_par[['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']])
    # test_.predict_class(path_classifier_int_maxdev_model='OUTPUT/Kept_RESULTS/Classification_result_may26_GOOD/classifier_resp_model.json',
    #                     integral_R=pred_integralR, max_deviation=pred_max_dev)
    # # print(pred_par)
    # params = pred_par.iloc[0].values
    # x = np.linspace(params[0], params[0]+70, 70)
    # R = response(x=x, par=params)
    # R_ideal = response_legacy(x=x, par=params)
    # plt.figure()
    # plt.stairs(h, label='data')
    # plt.plot(R, label='real')
    # plt.plot(R_ideal, label='ideal')
    # plt.title(f'integral tail = {pred_integralR}\n max deviation = {pred_max_dev}')
    # plt.legend()
    # plt.show()