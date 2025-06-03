import os, sys
import pandas as pd
import numpy as np
from BDT import split_train_test_dataset, BDT_Classifier, BDT_Regressor

if __name__=='__main__':
    # train_test split
    # root_path = 'data/labelledData'
    # split_train_test_dataset(path_to_data=root_path, frac_test=0.2) 20% of the whole dataset is used for testing

    #
    classifier = BDT_Classifier()
    # classifier.read_data(path_to_data='data/labelledData/train_valid')
    params = classifier.tune_hyperamaters()
    classifier.train(params=params)