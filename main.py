import os, sys
import pandas as pd
import numpy as np
from BDT import split_train_test_dataset, BDT_Classifier, BDT_Regressor

if __name__=='__main__':
    # train_test split
    # root_path = 'data/labelledData'
    # split_train_test_dataset(path_to_data=root_path, frac_test=0.2) 20% of the whole dataset is used for testing

    #
    # # CLASSIFICATION MODEL
    # path_to_data = 'data/labelledData'
    # output_path = 'OUTPUT/bdt'
    # classifier = BDT_Classifier(path_to_data=path_to_data, output_path=output_path)
    # params = classifier.tune_hyperamaters()
    # classifier.train(params=params)

    #
    # REGRESSION MODEL
    regressor = BDT_Regressor(path_to_data='data/labelledData', output_path='OUTPUT/bdt')
    params = regressor.tune_hyperparameters()
    regressor.train(params=params)
    regressor.classify_predicted_integralMaxdev(path_to_classifier_model='OUTPUT/bdt/classifier_bdt_model.json')