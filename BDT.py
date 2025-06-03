"""

"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

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


class BDT_Classifier:
    def __init__(self):
        self.input_columns = ['integral_R', 'max_deviation']
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes
        self.train_df = self.read_data(path_to_data='data/labelledData/train_valid')
        self.test_df = self.read_data(path_to_data='data/labelledData/test')
        self.classifier_model = self.model()

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
                                     n_iter=50, cv=5, verbose=0)
        classifier = rand_cv.fit(X=self.train_df[self.input_columns], y=self.train_df['class'])
        print('Best parameters : ', classifier.best_params_)
        return classifier.best_params_

    def train(self, params={}):
        self.test_df['class'] = self.test_df['class'].map(self.class_map)
        self.classifier_model.fit(params, X=self.train_df[self.input_columns], y=self.train_df['class'])
        predictions = self.classifier_model.predict(self.test_df[self.input_columns])
        print(predictions - self.test_df['class'])
        
class BDT_Regressor:
    def __init__(self):
        pass

    def model(self):
        pass

    def read_data(self):
        pass

    def tune_hyperparameters(self):
        pass