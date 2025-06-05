"""

"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LogNorm

def split_train_test_dataset(path_to_data=None, frac_test=0.2):
    """
        Split training and testing dataset:
            - read the dataset for each class
            - select frac_test of each class for the testing set and 1-frac_test for the training and validation
            - save the training/validation and testing dataset in the folders train_valid and test respectively
    """

    # try to create the output folders if they don't exist
    for d in ['train_valid', 'test']:
        try:
            os.mkdir('/'.join([path_to_data, d]))
        except:
            pass

    list_data = [f for f in os.listdir(path_to_data) if ('fit_results' in f) and ('.csv' in f)]
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

    # split tran/valid and test
    ## c1
    N_test_c1 = int(N_c1*frac_test)
    N_train_c1 = N_c1 - N_test_c1
    train_c1 = c1_df.iloc[:N_train_c1]
    test_c1 = c1_df.iloc[N_test_c1:N_c1]
    train_c1.to_csv('/'.join([path_to_data,'train_valid/train_c1.csv']), index=False)
    test_c1.to_csv('/'.join([path_to_data, 'test/test_c1.csv']), index=False)
    ## c2
    N_test_c2 = int(N_c2*frac_test)
    N_train_c2 = N_c2 - N_test_c2
    train_c2 = c2_df.iloc[:N_train_c2]
    test_c2 = c2_df.iloc[N_test_c2:N_c2]
    train_c2.to_csv('/'.join([path_to_data,'train_valid/train_c2.csv']), index=False)
    test_c2.to_csv('/'.join([path_to_data, 'test/test_c2.csv']), index=False)
    ## c3
    N_test_c3 = int(N_c3*frac_test)
    N_train_c3 = N_c3 - N_test_c3
    train_c3 = c3_df.iloc[:N_train_c3]
    test_c3 = c3_df.iloc[N_test_c3:N_c3]
    train_c3.to_csv('/'.join([path_to_data,'train_valid/train_c3.csv']), index=False)
    test_c3.to_csv('/'.join([path_to_data, 'test/test_c3.csv']), index=False)
    ## c4
    N_test_c4 = int(N_c4*frac_test)
    N_train_c4 = N_c4 - N_test_c4
    train_c4 = c4_df.iloc[:N_train_c4]
    test_c4 = c4_df.iloc[N_test_c4:N_c4]
    train_c4.to_csv('/'.join([path_to_data,'train_valid/train_c4.csv']), index=False)
    test_c4.to_csv('/'.join([path_to_data, 'test/test_c4.csv']), index=False)


def confusionMatrix(truth_Series, pred_Series, output_path, figname='confusionMatrix_onOutRegressor.png'):
    """
        Evaluation of the classification power.
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true=truth_Series, y_pred=pred_Series)
    columns = list(truth_Series.unique())

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
        self.test_df = self.read_data(path_to_data=f'{path_to_data}/test')
        self.classifier_model = self.model()
        self.iter_training = 0

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

    def tune_hyperamaters(self):
        model = XGBClassifier()
        objective = 'binary:logistic'
        # eval_metric = 'rmse'
        params = {
            'n_estimators': randint(10, 100),
            'max_depth': randint(3, 30),
            'max_leaves': randint(0, 30),
            'learning_rate': uniform(0.5, 0.5),
            'num_boost_round': randint(50, 300),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            # 'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }
        rand_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=3,
                                     n_iter=10, cv=3, verbose=True)
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
        # test on the testing dataset
        predictions = self.classifier_model.predict(test_df[self.input_columns])
        predicted_classes = [self.map_class[pred] for pred in predictions]
        test_df['prediction'] = predicted_classes
        test_df['class'] = test_df['class'].apply(lambda x: self.map_class[x])
        #
        # calculate the accuracy of the prediction
        accuracy = ((test_df['class']==test_df['prediction']).mean())*100
        print(f'Accuracy = {accuracy:.2f}%')
        # save the model if the accuracy >= 99.98%
        if (accuracy >= 99.85) or (self.iter_training >= 5):
            self.classifier_model.save_model(f'{self.output_path}/classifier_bdt_model.json')
        else:
            print('Accuracy not good enough to be saved.')
            params = self.tune_hyperamaters()
            self.train(params=params)
            self.iter_training += 1

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
            'n_estimators': randint(10, 100),
            'max_depth': randint(3, 30),
            'max_leaves': randint(0, 30),
            'learning_rate': uniform(0.5, 0.5),
            'num_boost_round': randint(50, 300),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'objective': [objective],
            'eval_metric': [eval_metric],
            'tree_method': ['hist'],
            'device': ['cuda']
        }
        model = XGBRegressor()
        rand_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=3,
                                     n_iter=50, cv=3, verbose=True)
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
        # test on the testing dataset
        predictions = self.regressor_model.predict(self.test_df[self.input_columns])
        predictions_df = pd.DataFrame(predictions, columns=[self.output_columns], index=self.test_df.index)
        predictions_df.columns = [f'pred_{c}' for c in self.output_columns]
        #
        # save the predicted values
        self.test_df = pd.concat([self.test_df, predictions_df], axis=1, join='outer')
        self.test_df.to_csv(f'{self.output_path}/predicted_integral_max_dev.csv', index=True)
        # 2d correlation plots for each class
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c1'], pred_df=self.test_df[self.test_df['class']=='c1'],
                   classname='c1', output_path=self.output_path)
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c2'], pred_df=self.test_df[self.test_df['class']=='c2'],
                   classname='c2', output_path=self.output_path)
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c3'], pred_df=self.test_df[self.test_df['class']=='c3'],
                   classname='c3', output_path=self.output_path)
        corr2dplot(truth_df=self.test_df[self.test_df['class']=='c4'], pred_df=self.test_df[self.test_df['class']=='c4'],
                   classname='c4', output_path=self.output_path)
        #
        # comparison between truth and prediction
        fig, ax = plt.subplots(1,2,figsize=(10*2, 10))
        ax[0].hist(predictions_df['pred_integral_R'], bins=100, histtype='step', label='prediction of the integral of tail')
        ax[0].hist(self.test_df['integral_R'], bins=100, histtype='step', label='true integral of the tail')
        ax[0].set_xlabel('Integral of the tail')
        ax[0].set_ylabel('#')
        ax[0].legend()
        ax[0].grid(True)
        #
        ax[1].hist(predictions_df['pred_max_deviation'], bins=100, histtype='step', label='prediction of the max deviation')
        ax[1].hist(self.test_df['max_deviation'], bins=100, histtype='step', label='true max deviation')
        ax[1].set_xlabel('max deviation')
        ax[1].set_ylabel('#')
        ax[1].legend()
        ax[1].grid(True)
        plt.savefig(f'{self.output_path}/comparison_truth_predictions_bdt.png')
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
        confusionMatrix(truth_Series=test_df['class'], pred_Series=test_df['pred_class'], output_path=self.output_path, figname='confusionMatrix_onOutRegressor.png')

class preClassifier:
    def __init__(self, path_to_fitparams_csv, path_to_WF_generated, output_path):
        self.path_to_fitparams_csv = path_to_fitparams_csv
        self.path_to_WF_generated = path_to_WF_generated
        self.output_path = output_path