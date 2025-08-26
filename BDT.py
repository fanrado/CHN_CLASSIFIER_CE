"""
    This script gathers the organized versions of the codes in Train_BDT.py and preClassifier.py.
    The class Sim_waveform, which is used to save the dataset into .npy files, is still in the file preClassifier.py.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LogNorm
from response import *

## This class Sim_waveform is a copy of the code in preClassifier
class Sim_waveform:
    '''
        Generate waveforms using the fit parameters and the response function.
    '''
    def __init__(self, path_to_sim=None, output_path=None, N_samples=3000):
        self.path_to_sim = path_to_sim
        self.output_path = output_path
        self.sim_data = pd.DataFrame()
        self.N_samples = N_samples
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
        nbins = 70
        nbins = 115
        x = np.linspace(params[0], params[0]+nbins, nbins)
        # 2us to tick unit => 2us/0.512
        params[2] = params[2]/0.512
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
        # N_samples = self.sim_data.shape[0]
        N_samples = self.N_samples
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
    
    def data2npy_torch(self):
        nbins = 70
        nbins = 115
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Get the number of samples to process
        N_samples = min(self.N_samples, len(self.sim_data))
        
        # Extract parameters for all samples at once
        params_df = self.sim_data[self.response_params].iloc[:N_samples]
        params_tensor = torch.tensor(params_df.values, dtype=torch.float32, device=device)
        
        # Create time array on GPU
        # Assuming params[0] is 't', we'll create x arrays for each sample
        t_values = params_tensor[:, 0]  # First column is 't'
        batch_size = params_tensor.shape[0]
        
        # Create x arrays: shape [batch_size, 70]
        x_offsets = torch.arange(nbins, dtype=torch.float32, device=device)
        x_arrays = t_values.unsqueeze(1) + x_offsets.unsqueeze(0)  # Broadcasting
        
        # Convert t_p from 2us to tick unit (divide by 0.512), only used for plotting``
        params_tensor[:, 2] = params_tensor[:, 2] / 0.512
        
        # Generate all waveforms at once using vectorized response function
        R_batch = response_torch(x_arrays, params_tensor)  # Shape: [batch_size, 70]
        params_tensor[:, 2] = params_tensor[:, 2] * 0.512

        # Move results back to CPU for numpy operations
        R_batch_cpu = R_batch.cpu().numpy()
        params_cpu = params_tensor.cpu().numpy()
        
        # Build enriched data list
        enriched_data = []
        for i in range(N_samples):
            # Create parameter dictionary
            dict_params = dict(zip(self.response_params, params_cpu[i]))
            
            # Add additional data
            dict_params['#Ch.#'] = self.sim_data['#Ch.#'].iloc[i]
            dict_params['class'] = self.sim_data['class'].iloc[i]
            dict_params['wf'] = R_batch_cpu[i]
            dict_params['integral_R'] = self.sim_data['integral_R'].iloc[i]
            dict_params['max_deviation'] = self.sim_data['max_deviation'].iloc[i]
            
            enriched_data.append(dict_params)
        
        # Save the enriched_data to a .npy file
        output_file = self.path_to_sim.split('/')[-1].split('.')[0]
        output_file = f'{self.output_path}/{output_file}.npy'
        np.save(output_file, enriched_data)
        print(f'Data saved to {output_file}')
        
        # Clean up GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def split_train_test_dataset(path_to_data=None, output_path=None, frac_test=0.2, N1=None, N2=None, N3=None, N4=None):
    """
        Split training and testing dataset:
            - read the dataset for each class
            - select frac_test of each class for the testing set and 1-frac_test for the training and validation
            - save the training/validation and testing dataset in the folders train_valid and test respectively
    """

    # try to create the output folders if they don't exist
    try:
        os.mkdir(output_path)
    except:
        pass
    for d in ['train_valid', 'test']:
        try:
            os.mkdir('/'.join([output_path, d]))
        except:
            pass

    # list_data = [f for f in os.listdir(path_to_data) if ('fit_results' in f) and ('.csv' in f)]
    list_data = [f for f in os.listdir(path_to_data) if '.csv' in f]
    # read files and concatenate in one dataframe
    df = pd.DataFrame()
    for i, f in enumerate(list_data):
        tmpdf = pd.read_csv('/'.join([path_to_data, f]))
        if i==0:
            df = tmpdf.copy()
        else:
            df = pd.concat([df, tmpdf], axis=0)
    # select classes
    c1_df = df[df['class']=='c1']
    c2_df = df[df['class']=='c2']
    c3_df = df[df['class']=='c3']
    c4_df = df[df['class']=='c4']

    # Total numbers of samples in each class
    N_c1 = len(c1_df)
    N_c2 = len(c2_df)
    N_c3 = len(c3_df)
    N_c4 = len(c4_df)
    if (N1 is not None) and (N2 is not None) and (N3 is not None) and (N4 is not None):
        N_c1, N_c2, N_c3, N_c4 = N1, N2, N3, N4


    # split tran/valid and test
    ## c1
    N_test_c1 = int(N_c1*frac_test)
    N_train_c1 = N_c1 - N_test_c1
    # print(N_train_c1, N_test_c1)
    # sys.exit()
    train_c1 = c1_df.iloc[:N_train_c1]
    test_c1 = c1_df.iloc[N_train_c1:N_c1]
    train_c1.to_csv('/'.join([output_path,'train_valid/train_c1.csv']), index=False)
    test_c1.to_csv('/'.join([output_path, 'test/test_c1.csv']), index=False)
    ## c2
    N_test_c2 = int(N_c2*frac_test)
    N_train_c2 = N_c2 - N_test_c2
    train_c2 = c2_df.iloc[:N_train_c2]
    test_c2 = c2_df.iloc[N_train_c2:N_c2]
    train_c2.to_csv('/'.join([output_path,'train_valid/train_c2.csv']), index=False)
    test_c2.to_csv('/'.join([output_path, 'test/test_c2.csv']), index=False)
    ## c3
    N_test_c3 = int(N_c3*frac_test)
    N_train_c3 = N_c3 - N_test_c3
    train_c3 = c3_df.iloc[:N_train_c3]
    test_c3 = c3_df.iloc[N_train_c3:N_c3]
    train_c3.to_csv('/'.join([output_path,'train_valid/train_c3.csv']), index=False)
    test_c3.to_csv('/'.join([output_path, 'test/test_c3.csv']), index=False)
    ## c4
    N_test_c4 = int(N_c4*frac_test)
    N_train_c4 = N_c4 - N_test_c4
    train_c4 = c4_df.iloc[:N_train_c4]
    test_c4 = c4_df.iloc[N_train_c4:N_c4]
    train_c4.to_csv('/'.join([output_path,'train_valid/train_c4.csv']), index=False)
    test_c4.to_csv('/'.join([output_path, 'test/test_c4.csv']), index=False)

from sklearn.metrics import classification_report

def confusionMatrix(truth_Series, pred_Series, output_path, figname='confusionMatrix_onOutRegressor.png'):
    """
        Evaluation of the classification power.
    """
    try:
        os.mkdir(output_path)
    except:
        pass
    
    # validation of the confusion matrix
    # report = classification_report(y_true=truth_Series, y_pred=pred_Series, target_names=['c1', 'c2', 'c3', 'c4'])
    # print("Classification report:")
    # print(report)
    # Create confusion matrix
    target_names=['c1', 'c2', 'c3', 'c4']
    cm = confusion_matrix(y_true=truth_Series, y_pred=pred_Series)
    # columns = list(truth_Series.unique())
    # print(cm)
    # print(columns)
    # print(list(pred_Series.unique()))
    # sys.exit()
    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    # Create heatmap
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='crest',
                xticklabels=target_names,
                yticklabels=target_names)

    # Add labels
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')

    # Show plot
    plt.tight_layout()
    plt.savefig(f'{output_path}/{figname}')
    plt.close()

def corr2dplot(truth_df, pred_df, output_path, classname='c1'):
    fig, ax = plt.subplots(1,2, figsize=(8*2, 8))
    h1 = ax[0].hist2d(pred_df['pred_integral_R'], truth_df['integral_R'], 
            bins=30, norm=LogNorm(),cmap='viridis')
    cbar1 = plt.colorbar(h1[3])
    cbar1.set_label('Counts')
    ax[0].set_xlabel('predicted integral_R')
    ax[0].set_ylabel('true integral_R')
    ax[0].set_title(classname)
    
    h2 = ax[1].hist2d(pred_df['pred_max_deviation'], truth_df['max_deviation'], 
            bins=30, norm=LogNorm(),cmap='viridis')
    cbar2 = plt.colorbar(h2[3])
    cbar2.set_label('Counts')
    ax[1].set_xlabel('predicted max_deviation')
    ax[1].set_ylabel('true max_deviation')
    ax[1].set_title(classname)

    plt.tight_layout()
    plt.savefig(f'{output_path}/{classname}_2dCorr.png')
    plt.close()

def compare_metrics(data_df, output_path):
    # comparison between truth and prediction
    fig, ax = plt.subplots(1,2,figsize=(10*2, 10))
    ax[0].hist(data_df['pred_integral_R'], bins=100, linewidth=3, histtype='step', label='prediction of the integral of tail')
    ax[0].hist(data_df['integral_R'], bins=100, linewidth=3, histtype='step', label='true integral of the tail')
    ax[0].set_xlabel('Integral of the tail')
    ax[0].set_ylabel('#')
    ax[0].legend()
    ax[0].grid(True)
    #
    ax[1].hist(data_df['pred_max_deviation'], bins=100, linewidth=3, histtype='step', label='prediction of the max deviation')
    ax[1].hist(data_df['max_deviation'], bins=100, linewidth=3, histtype='step', label='true max deviation')
    ax[1].set_xlabel('max deviation')
    ax[1].set_ylabel('#')
    ax[1].legend()
    ax[1].grid(True)
    plt.savefig(f'{output_path}/comparison_truth_predictions_bdt.png')
    plt.close()

class BDT_Classifier:
    """
        This class is used to train and test the BDT classification model.
    """
    def __init__(self, path_to_data=None, output_path=None):
        self.output_path = output_path
        try:
            os.mkdir(self.output_path)
        except:
            pass
        self.input_columns = ['integral_R', 'max_deviation']
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes
        self.train_df = self.read_data(path_to_data=f'{path_to_data}/train_valid')
        self.test_df = self.read_data(path_to_data=f'{path_to_data}/test') # not used during training
        self.classifier_model = self.model()
        self.iter_training = 0
        self.n_iter = 10

    def model(self):
        classifier_model = XGBClassifier()
        return classifier_model

    def read_data(self, path_to_data=None):
        list_files = [f for f in os.listdir(path_to_data)]
        df = pd.DataFrame()
        for i, f in enumerate(list_files):
            tmpdf = pd.read_csv('/'.join([path_to_data, f]))
            if i==0:
                df = tmpdf.copy()
            else:
                df = pd.concat([df, tmpdf], axis=0)
        # shuffle the dataframe rows
        df = df.sample(frac=1)
        df['class'] = df['class'].map(self.class_map)
        # print(df)
        return df

    def tune_hyperamaters(self, n_iter=10):
        model = XGBClassifier()
        # objective = 'binary:logistic'
        objective = 'multi:softprob'
        # eval_metric = 'rmse'
        params = {
            'num_class': [4],
            'n_estimators': randint(100, 200),
            'max_depth': randint(15,20),
            'max_leaves': randint(0, 30),
            'learning_rate': uniform(0.4, 0.3),
            'num_boost_round': randint(100, 300),
            'min_child_weight': randint(15, 20),
            'subsample': uniform(0.8, 0.1),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            # 'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }
        rand_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=4,
                                     n_iter=n_iter, cv=3, verbose=True)
        classifier = rand_cv.fit(X=self.train_df[self.input_columns], y=self.train_df['class'])
        print('Best parameters : ', classifier.best_params_)
        return classifier.best_params_

    def train(self, params={}):
        self.classifier_model.set_params(**params)
        test_df = self.test_df.copy()
        # split the training set to train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(self.train_df[self.input_columns], self.train_df['class'], test_size=0.2, random_state=32)
        # training the classifier
        self.classifier_model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)
        #
        # test on the TESTING dataset
        predictions = self.classifier_model.predict(test_df[self.input_columns])
        predicted_classes = [self.map_class[pred] for pred in predictions]
        test_df['prediction'] = predicted_classes
        test_df['class'] = test_df['class'].apply(lambda x: self.map_class[x])
        #
        # calculate the accuracy of the prediction
        accuracy = ((test_df['class']==test_df['prediction']).mean())*100
        print(f'Accuracy = {accuracy:.2f}%')
        # save the model if the accuracy >= 99.98%
        # if (accuracy >= 99.85) or (self.iter_training >= 5):
        self.classifier_model.save_model(f'{self.output_path}/classifier_bdt_model.json')
        # evaluate the classification power
        confusionMatrix(truth_Series=test_df['class'], pred_Series=test_df['prediction'], output_path=f'{self.output_path}/prediction_testdataset', figname='confusionMatrix_calculatedMetrics.png')
        # else:
        #     print('Accuracy not good enough to be saved.')
        #     self.n_iter += 1
        #     params = self.tune_hyperamaters(n_iter=self.n_iter)
        #     self.train(params=params)
        #     self.iter_training += 1
        #     print(f'ITER_TRAINING = {self.iter_training}')

class BDT_Regressor:
    """
        This class is used to train the BDT regression model.
    """
    def __init__(self, path_to_data=None, output_path=None):
        self.output_path = output_path
        try:
            os.mkdir(self.output_path)
        except:
            pass
        self.input_columns = ['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.output_columns = ['integral_R', 'max_deviation']
        self.train_df = self.read_data(path_to_data=f'{path_to_data}/train_valid')
        self.test_df = self.read_data(path_to_data=f'{path_to_data}/test')
        self.regressor_model = self.model()
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes

    def model(self):
        regressor_model = XGBRegressor()
        return regressor_model

    def read_data(self, path_to_data=None):
        list_files = [f for f in os.listdir(path_to_data)]
        df = pd.DataFrame()
        for i, f in enumerate(list_files):
            tmpdf = pd.read_csv('/'.join([path_to_data, f]))
            if i==0:
                df = tmpdf.copy()
            else:
                df = pd.concat([df, tmpdf])
        # shuffle the dataframe rows
        df = df.sample(frac=1)
        return df

    def tune_hyperparameters(self):
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        params = {
            'n_estimators': randint(100, 200),
            'max_depth': randint(15,20),
            'max_leaves': randint(0, 30),
            'learning_rate': uniform(0.4, 0.3),
            'num_boost_round': randint(100, 300),
            'min_child_weight': randint(15, 20),
            'subsample': uniform(0.8, 0.1),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }
        model = XGBRegressor()
        rand_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=4,
                                     n_iter=10, cv=3, verbose=True)
        regressor = rand_cv.fit(X=self.train_df[self.input_columns], y=self.train_df[self.output_columns])
        print('Best parameters : ', regressor.best_params_)
        return regressor.best_params_
    
    def train(self, params={}):
        self.regressor_model.set_params(**params)
        # test_df = self.test_df.copy()
        # split the training set to train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(self.train_df[self.input_columns], self.train_df[self.output_columns], test_size=0.2, random_state=32)
        # training the regressor
        self.regressor_model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)
        #
        # test on the TESTING dataset
        predictions = self.regressor_model.predict(self.test_df[self.input_columns])
        predictions_df = pd.DataFrame(predictions, columns=[self.output_columns], index=self.test_df.index)
        predictions_df.columns = [f'pred_{c}' for c in self.output_columns]
        #
        # save the predicted values
        self.test_df = pd.concat([self.test_df, predictions_df], axis=1, join='outer')
        self.test_df.to_csv(f'{self.output_path}/prediction_testdataset/predicted_integral_max_dev_onTestdataset.csv', index=True)
        # 2d correlation plots for each class
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c1'], pred_df=self.test_df[self.test_df['class']=='c1'],
                   classname='c1', output_path=f'{self.output_path}/prediction_testdataset')
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c2'], pred_df=self.test_df[self.test_df['class']=='c2'],
                   classname='c2', output_path=f'{self.output_path}/prediction_testdataset')
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c3'], pred_df=self.test_df[self.test_df['class']=='c3'],
                   classname='c3', output_path=f'{self.output_path}/prediction_testdataset')
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c4'], pred_df=self.test_df[self.test_df['class']=='c4'],
                   classname='c4', output_path=f'{self.output_path}/prediction_testdataset')
        #
        # comparison between truth and prediction
        fig, ax = plt.subplots(1,2,figsize=(10*2, 10))
        ax[0].hist(predictions_df['pred_integral_R'], bins=100, linewidth=3, histtype='step', label='prediction of the integral of tail')
        ax[0].hist(self.test_df['integral_R'], bins=100, linewidth=3, histtype='step', label='true integral of the tail')
        ax[0].set_xlabel('Integral of the tail')
        ax[0].set_ylabel('#')
        ax[0].legend()
        ax[0].grid(True)
        #
        ax[1].hist(predictions_df['pred_max_deviation'], bins=100, linewidth=3, histtype='step', label='prediction of the max deviation')
        ax[1].hist(self.test_df['max_deviation'], bins=100, linewidth=3, histtype='step', label='true max deviation')
        ax[1].set_xlabel('max deviation')
        ax[1].set_ylabel('#')
        ax[1].legend()
        ax[1].grid(True)
        plt.savefig(f'{self.output_path}/prediction_testdataset/comparison_truth_predictions_bdt.png')
        plt.close()
        #
        # Save the model
        self.regressor_model.save_model(f'{self.output_path}/regressor_bdt_model.json')

    def classify_predicted_integralMaxdev(self, path_to_classifier_model=None):
        classifier_model = XGBClassifier()
        classifier_model.load_model(path_to_classifier_model)
        test_df = self.test_df.copy()
        test_df.drop(columns=self.output_columns, axis=1, inplace=True)
        for c in self.output_columns:
            test_df[c] = test_df[f'pred_{c}']
            test_df.drop(columns=f'pred_{c}', axis=1, inplace=True)
        predictions = classifier_model.predict(test_df[self.output_columns])
        pred_df = pd.DataFrame(predictions, columns=['pred_class'], index=test_df.index)
        pred_df['pred_class'] = pred_df['pred_class'].apply(lambda x: self.map_class[x])
        test_df = pd.concat([test_df, pred_df], axis=1, join='outer')
        # evaluate the classification power
        confusionMatrix(truth_Series=test_df['class'], pred_Series=test_df['pred_class'], output_path=f'{self.output_path}/prediction_testdataset', figname='confmat_classify_predictedmetrics.png')
        test_df.to_csv(f'{self.output_path}/prediction_testdataset/classified_predicted_integral_max_dev_onTestdataset.csv', index=True)

class Classify:
    def __init__(self, regressor_model=None, classifier_model=None, path_to_data=None, output_path=None):
        self.regressor_model    = XGBRegressor()
        self.regressor_model.load_model(regressor_model)
        self.classifier_model   = XGBClassifier()
        self.classifier_model.load_model(classifier_model)
        self.path_to_data       = path_to_data
        self.output_path        = output_path
        self.data               = pd.DataFrame()
        self.colsInput_regressor    = ['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.colsOutput_regressor   = ['integral_R', 'max_deviation']
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes
    
    def read_data(self, key_in_name='fit_results', file_ext='.csv', sep=','):
        list_files = [f for f in os.listdir(self.path_to_data) if (key_in_name in f) and (file_ext in f)]
        data_df = pd.DataFrame()
        for i, f in enumerate(list_files):
            tmp_df = pd.read_csv('/'.join([self.path_to_data, f]), sep=sep)
            if i==0:
                data_df = tmp_df.copy()
            else:
                data_df = pd.concat([data_df, tmp_df.copy()], axis=0)
        return data_df

    def predMetrics(self, data_df=None, plot2dcorr=False):
        # Predict the integral of the tail and the maximum deviation between the tails of the ideal and real responses
        predictions = self.regressor_model.predict(data_df[self.colsInput_regressor])
        predictions_df = pd.DataFrame(predictions, columns=[self.colsOutput_regressor], index=data_df.index)
        predictions_df.columns = [f'pred_{c}' for c in self.colsOutput_regressor]
        data_with_pred_df = pd.concat([data_df, predictions_df], axis=1, join='outer')
        if plot2dcorr:
            if 'class' not in list(data_with_pred_df.columns):
                print("Cannot generate the correlation plot. The data is not labeled.")
                return data_with_pred_df
            # 2d correlation plots for each class
            corr2dplot(truth_df=data_with_pred_df[data_with_pred_df['class']=='c1'], pred_df=data_with_pred_df[data_with_pred_df['class']=='c1'],
                    classname='c1', output_path=self.output_path)
            corr2dplot(truth_df=data_with_pred_df[data_with_pred_df['class']=='c2'], pred_df=data_with_pred_df[data_with_pred_df['class']=='c2'],
                    classname='c2', output_path=self.output_path)
            corr2dplot(truth_df=data_with_pred_df[data_with_pred_df['class']=='c3'], pred_df=data_with_pred_df[data_with_pred_df['class']=='c3'],
                    classname='c3', output_path=self.output_path)
            corr2dplot(truth_df=data_with_pred_df[data_with_pred_df['class']=='c4'], pred_df=data_with_pred_df[data_with_pred_df['class']=='c4'],
                    classname='c4', output_path=self.output_path)
            # compare the metrics: predicted vs calculated
            compare_metrics(data_df=data_with_pred_df, output_path=self.output_path)
        return data_with_pred_df
    
    def classify_metrics(self, data_df=None, plotconfmatrix=False):
        data_df.drop(columns=self.colsOutput_regressor, axis=1, inplace=True)
        for c in self.colsOutput_regressor:
            data_df[c] = data_df[f'pred_{c}']
            data_df.drop(columns=f'pred_{c}', axis=1, inplace=True)
        predictions = self.classifier_model.predict(data_df[self.colsOutput_regressor])
        pred_df = pd.DataFrame(predictions, columns=['pred_class'], index=data_df.index)
        pred_df['pred_class'] = pred_df['pred_class'].apply(lambda x: self.map_class[x])
        test_df = pd.concat([data_df, pred_df], axis=1, join='outer')
        if plotconfmatrix:
            # evaluate the classification power
            confusionMatrix(truth_Series=test_df['class'], pred_Series=test_df['pred_class'], output_path=self.output_path, figname='confusionMatrix_onOutRegressor.png')
        return test_df
    
    def run_classification(self, info_data_dict={
                                                    'key_in_name'   : 'fit_results',
                                                    'file_ext'      : '.csv',
                                                    'sep'           : ','
                                                },
                                plot2dcorr=False, plotconfmatrix=False):
        data_df = self.read_data(key_in_name=info_data_dict['key_in_name'], file_ext=info_data_dict['file_ext'], sep=info_data_dict['sep'])
        predicted_metrics_df = self.predMetrics(data_df=data_df, plot2dcorr=plot2dcorr)
        predicted_metrics_df.to_csv(f'{self.output_path}/predicted_integral_max_dev_ondata.csv', index=True)
        classified_df = self.classify_metrics(data_df=predicted_metrics_df, plotconfmatrix=plotconfmatrix)
        classified_df.to_csv(f'{self.output_path}/classified_data.csv', index=True)

# # Preclassification
class preClassifier:
    def __init__(self, path_to_train, output_path):
        self.path_to_train  = path_to_train
        self.output_path    = output_path
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes
        self.input_data     = self.read_npy()

    def read_npy(self):
        npy2list = []
        for f in os.listdir(self.path_to_train):
            tmpdata = np.load('/'.join([self.path_to_train, f]), allow_pickle=True).tolist()
            npy2list += tmpdata
        data_df = self.npylist2df(npylist=npy2list)
        return data_df
    
    def npylist2df(self, npylist):
        data_df = pd.DataFrame()
        for i, sample in enumerate(npylist):
            npylist[i]['sample_id'] = i
        data_df = pd.DataFrame(npylist)
        data_df = data_df.sample(frac=1)
        data_df['class'] = data_df['class'].map(self.class_map)
        return data_df

    def split_train_test_valid(self, data_df):
        X = np.array(data_df['wf'].tolist())
        y = data_df['class'].values
        sample_ids = data_df['sample_id'].values
        # train/test with frac_test = 0.2
        X_train_tmp, X_test, y_train_tmp, y_test, ids_train_tmp, ids_test = train_test_split(X, y, sample_ids, test_size=0.2, random_state=42)
        # # train/valid with frac_valid = 0.2
        # X_train, X_valid, y_train, y_valid, ids_train, ids_valid = train_test_split(X_train_tmp, y_train_tmp, ids_train_tmp, test_size=0.2, random_state=42)
        return {
            "X_train": X_train_tmp,
            "y_train": y_train_tmp,
            "ids_train": ids_train_tmp,
            # "X_valid": X_valid,
            # "y_valid": y_valid,
            # "ids_valid": ids_valid,
            "X_test": X_test,
            "y_test": y_test,
            "ids_test": ids_test
        }

    def tune_hyperparameters(self, X_train, y_train):
        model = XGBClassifier()
        # objective = 'binary:logistic'
        objective = 'multi:softprob'
        # eval_metric = 'rmse'
        params = {
            'num_class': [4],
            'n_estimators': randint(100, 200),
            'max_depth': randint(15,20),
            'max_leaves': randint(0, 30),
            'learning_rate': uniform(0.4, 0.3),
            'num_boost_round': randint(100, 300),
            'min_child_weight': randint(15, 20),
            'subsample': uniform(0.8, 0.1),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            # 'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }
        rand_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=4,
                                     n_iter=10, cv=3, verbose=True)
        classifier = rand_cv.fit(X=X_train, y=y_train)
        print('Best parameters : ', classifier.best_params_)
        return classifier.best_params_

    def train(self, best_params, X_set, y_set, ids_set):
        model = XGBClassifier()
        model.set_params(**best_params)
        # split training dataset into train/valid with frac_valid = 0.2
        X_train, X_valid, y_train, y_valid, ids_train, ids_valid = train_test_split(X_set, y_set, ids_set, test_size=0.2, random_state=42)
        model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)
        return model

    def plot_wf_with_class(self, wf_data, output_path='', trueclass='', predclass=''):
        posmax = np.argmax(wf_data.reshape(-1,1))
        maxamp = wf_data[posmax]
        fig, ax = plt.subplots()
        rect = patches.Rectangle((posmax+4, -int(maxamp/6)), 70-posmax-4-1, maxamp/3, linewidth=2, edgecolor='red', facecolor='blue', alpha=0.1)
        ax.plot(wf_data, label=f'true class : {trueclass}; pred class : {predclass}')
        ax.add_patch(rect)
        ax.grid(True)
        ax.legend()
        fig.savefig(output_path)
        plt.close()

    def run(self):
        # split the dataset into train and test
        splitted_dataset = self.split_train_test_valid(data_df=self.input_data.copy())
        X_train, y_train, ids_train = splitted_dataset['X_train'], splitted_dataset['y_train'], splitted_dataset['ids_train']
        X_test, y_test, ids_test = splitted_dataset['X_test'], splitted_dataset['y_test'], splitted_dataset['ids_test']
        #
        # find best hyperparameters
        best_params = self.tune_hyperparameters(X_train=X_train, y_train=y_train)
        model = self.train(best_params=best_params, X_set=X_train, y_set=y_train, ids_set=ids_train)
        model.save_model(f'{self.output_path}/preclassifier/preclassifier.json')
        #
        # classification of the testing dataset
        # predictions = model.predict(X_test[0].reshape(1,-1))
        predictions = model.predict(X_test)
        predicted_classes = [self.map_class[pred] for pred in predictions]
        # print(predicted_classes)
        y_df = pd.DataFrame({'pred_class': predicted_classes, 'true_class': y_test, 'sample_id': ids_test})
        y_df['true_class'] = y_df['true_class'].apply(lambda x: self.map_class[x])
        confusionMatrix(truth_Series=y_df['true_class'], pred_Series=y_df['pred_class'], output_path=f'{self.output_path}/preclassifier', figname='preclassified_wf.png') # THIS RESULT IS TOO GOOD TO BE TRUE
        #
        # calculate the accuracy of the prediction
        accuracy = ((y_df['true_class']==y_df['pred_class']).mean())*100
        print(f'Accuracy = {accuracy:.2f}%')
        #
        # print(self.input_data)
        # print(len(self.input_data['sample_id']), len(ids_test))
        # print(self.input_data.loc[ids_test])
        test_df = self.input_data.loc[ids_test]
        # enriched_test_df = pd.concat([test_df, y_df], axis=1, join='outer')
        enriched_test_df = test_df.merge(y_df, how='outer', on='sample_id')
        enriched_test_df.to_csv(f'{self.output_path}/preclassifier/preclassification_testdataset.csv', index=False)
    
    def generate_wf_with_class(self, enriched_test_df):
        c1_df = enriched_test_df[enriched_test_df['true_class']=='c1'].copy().reset_index().drop('index', axis=1)
        c1_df = c1_df.iloc[:100].copy().reset_index().drop('index',axis=1)
        X = np.array(c1_df['wf'].tolist())
        ytrue = c1_df['true_class'].tolist()
        ypred = c1_df['pred_class'].tolist()
        chn = c1_df['#Ch.#'].tolist()
        for id in list(c1_df.index):
            self.plot_wf_with_class(wf_data=X[id], output_path=f'{self.output_path}/preclassifier/testPreclassifier/chresp_TRUE{ytrue[id]}_PRED{ypred[id]}_chn{chn[id]}.png', trueclass=ytrue[id], predclass=ypred[id])
        c2_df = enriched_test_df[enriched_test_df['true_class']=='c2'].copy().reset_index().drop('index', axis=1)
        c2_df = c2_df.iloc[:100].copy().reset_index().drop('index',axis=1)
        X = np.array(c2_df['wf'].tolist())
        ytrue = c2_df['true_class'].tolist()
        ypred = c2_df['pred_class'].tolist()
        chn = c2_df['#Ch.#'].tolist()
        for id in list(c2_df.index):
            self.plot_wf_with_class(wf_data=X[id], output_path=f'{self.output_path}/preclassifier/testPreclassifier/chresp_TRUE{ytrue[id]}_PRED{ypred[id]}_chn{chn[id]}.png', trueclass=ytrue[id], predclass=ypred[id])
        c3_df = enriched_test_df[enriched_test_df['true_class']=='c3'].copy().reset_index().drop('index', axis=1)
        c3_df = c3_df.iloc[:100].copy().reset_index().drop('index',axis=1)
        X = np.array(c3_df['wf'].tolist())
        ytrue = c3_df['true_class'].tolist()
        ypred = c3_df['pred_class'].tolist()
        chn = c3_df['#Ch.#'].tolist()
        for id in list(c3_df.index):
            self.plot_wf_with_class(wf_data=X[id], output_path=f'{self.output_path}/preclassifier/testPreclassifier/chresp_TRUE{ytrue[id]}_PRED{ypred[id]}_chn{chn[id]}.png', trueclass=ytrue[id], predclass=ypred[id])
        c4_df = enriched_test_df[enriched_test_df['true_class']=='c4'].copy().reset_index().drop('index', axis=1)
        c4_df = c4_df.iloc[:100].copy().reset_index().drop('index',axis=1)
        X = np.array(c4_df['wf'].tolist())
        ytrue = c4_df['true_class'].tolist()
        ypred = c4_df['pred_class'].tolist()
        chn = c4_df['#Ch.#'].tolist()
        for id in list(c4_df.index):
            self.plot_wf_with_class(wf_data=X[id], output_path=f'{self.output_path}/preclassifier/testPreclassifier/chresp_TRUE{ytrue[id]}_PRED{ypred[id]}_chn{chn[id]}.png', trueclass=ytrue[id], predclass=ypred[id])

import uproot
class TestPreclassifier:
    """
        This class will be used for the following tasks:
            - open a root file.
            - Get one 1d histogram of the channel response and store it in a numpy array.
            - Without passing through the prediction of the fit parameters, predict the class of the input waveform directly.
    """
    def __init__(self, path_to_root_file='', hist_prefix='hist_0', output_path=''):
        self.nbins              = 115 # 70
        self.path_to_root_file  = path_to_root_file
        self.run_number         = int(path_to_root_file.split('run_')[-1].split('.')[0])
        self.output_path        = output_path
        self.hist_prefix        = hist_prefix
        self.root_data, self.all_channels          = self.read_ROOT(filename=self.path_to_root_file)
        # print(self.all_channels)
        self.chn_response       = None
        self.predictions        = None
        # self.__fit_params     = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.class_map          = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class          = {v: k for k, v in self.class_map.items()} # numbers to classes
        self.class_to_meaning   = {'c1': 'Undershoot only',
                                   'c2': 'Undershoot with small overshoot',
                                   'c3': 'Overshoot with small undershoot',
                                   'c4': 'Overshoot only'}
    
    def read_ROOT(self, filename):
        root_file = uproot.open(filename)
        all_hist1d = [f for f in root_file.keys() if self.hist_prefix in f]
        all_channels = [int(hist1d.split(';')[0].split('_')[-1]) for hist1d in all_hist1d]
        return root_file, all_channels
    
    def get_histogram(self, root_data, histname, Npoints=70):
        TH1D_hist = root_data[histname].to_numpy()[0][:Npoints]
        return TH1D_hist
    
    def getCHN_resp(self, chn):
        histname = '_'.join([self.hist_prefix, 'channel', f'{chn};1'])
        hist = self.get_histogram(root_data=self.root_data, histname=histname, Npoints=self.nbins)
        wf = hist.copy().reshape(1,-1)
        return wf, hist # wf is the input of the preclassifier BDT; hist is the original waveform.
    
    def load_bdt_model(self, path_to_model=''):
        preclassifier_model = XGBClassifier()
        preclassifier_model.load_model(path_to_model)
        return preclassifier_model

    def predict_oneCHN(self, path_to_model='', chn=0, savefig=False):
        if savefig:
            try:
                os.mkdir(f'{self.output_path}/testPreclassifier_fromROOT')
            except:
                pass
        # # read channel response from root file
        wf, hist = self.getCHN_resp(chn=chn)
        # # load bdt classifier model
        preClassifier_model = self.load_bdt_model(path_to_model=path_to_model)
        # # predict the class of the waveform : positive peak
        predictions = preClassifier_model.predict(wf)
        predicted_class = self.map_class[predictions[0]]
        # print(f'Predicted class : {predicted_class}')

        if savefig:
            # print(self.map_class[predictions[0]])
            posmax = np.argmax(wf.reshape(-1,1))
            # print(70-posmax-4)
            fig,ax = plt.subplots()
            # rect = patches.Rectangle((posmax+4, -500), 70-posmax-4-1, 2000, linewidth=2, edgecolor='red', facecolor='blue', alpha=0.1)
            ax.plot(hist, label=f'pred class = {self.map_class[predictions[0]]}')
            ax.legend()
            # ax.add_patch(rect)
            ax.grid()
            fig.savefig(f'{self.output_path}/testPreclassifier_fromROOT/wf_{self.hist_prefix}_chn{chn}_{self.map_class[predictions[0]]}.png')
            plt.close()
            # sys.exit()
            if self.map_class[predictions[0]]=='c3':
                print(self.map_class[predictions[0]])

        return {'chn': chn, 'class': predicted_class, 'class_meaning': self.class_to_meaning[predicted_class]}
    
    def run(self, path_to_model='', savefig=False, Nchannels=2000):
        all_chn_results = {'chn': [], 'class': [], 'class_meaning': []}
        for chn in self.all_channels:
            onechn_pred = self.predict_oneCHN(path_to_model=path_to_model, chn=chn, savefig=savefig)
            all_chn_results['chn'].append(onechn_pred['chn'])
            all_chn_results['class'].append(onechn_pred['class'])
            all_chn_results['class_meaning'].append(onechn_pred['class_meaning'])
            if chn%100==0:
                print(f'{100*chn/Nchannels:.2f}% of {Nchannels}')
            if chn == Nchannels:
                break
        prediction_df = pd.DataFrame(all_chn_results)
        prediction_df.to_csv(f'{self.output_path}/preclassification_ROOT_{self.hist_prefix}_run_{self.run_number}.csv', index=False)

import scipy.signal as signal
class ToyTestPreclassifier:
    def __init__(self, peaktype='Positive', rootfilename='', output_path=''):
        self.output_path    = output_path
        self.peaktype       = peaktype
        self.run_number     = int(rootfilename.split('-')[1])
        self.rootdata       = uproot.open(rootfilename)
        self.h_daq          = self.rootdata['h_daq'].to_numpy()[0]
        self.all_channels   = [chn for chn in range(len(self.h_daq))]
        self.class_map      = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class      = {v: k for k, v in self.class_map.items()} # numbers to classes
    
    def gethistogram1d(self, chn):
        test_function = False
        # this is assuming positive peak
        # n_before_peak = 4
        # n_after_peak = 66
        n_before_peak = 15
        n_after_peak = 100
        onehist1d = self.h_daq[chn]
        pospeaks = signal.find_peaks(onehist1d, height=0.9*np.max(onehist1d))
        pospeak_of_interest = pospeaks[0][0]
        wf = onehist1d[pospeak_of_interest-n_before_peak:pospeak_of_interest+n_after_peak]
        if test_function:
            plt.figure()
            plt.plot(wf)
            plt.grid(True)
            plt.savefig(f'{self.output_path}/test_plot/test_chn{chn}.png')
            plt.close()
        return chn, wf.reshape(1, -1)
    
    def load_bdt_model(self, path_to_model=''):
        preclassifier_model = XGBClassifier()
        preclassifier_model.load_model(path_to_model)
        return preclassifier_model
    
    def predict_oneCHN(self, path_to_model='', chn=0):
        # # read channel response from root file
        chn, wf = self.gethistogram1d(chn=chn)
        # # load bdt classifier model
        preClassifier_model = self.load_bdt_model(path_to_model=path_to_model)
        # # predict the class of the waveform : positive peak
        predictions = preClassifier_model.predict(wf)
        return {'chn': chn, 'class': self.map_class[predictions[0]]}
    
    def run(self, path_to_model='', Nchannels=2000):
        all_chn_results = {'chn': [], 'class': []}
        for chn in self.all_channels:
            onechn_pred = self.predict_oneCHN(path_to_model=path_to_model, chn=chn)
            all_chn_results['chn'].append(onechn_pred['chn'])
            all_chn_results['class'].append(onechn_pred['class'])
            if chn%100==0:
                print(f'{100*chn/Nchannels:.2f}% of {Nchannels}')
            # if chn == Nchannels:
            #     break
        prediction_df = pd.DataFrame(all_chn_results)
        prediction_df.to_csv(f'{self.output_path}/preclassification_ROOT_run_{self.run_number}.csv', index=False)