import os, sys
import pandas as pd
import numpy as np
from BDT import split_train_test_dataset, BDT_Classifier, BDT_Regressor, Classify, preClassifier, TestPreclassifier

if __name__=='__main__':
    # train_test split
    # root_path = 'data/labelledData/labelledData'
    # split_train_test_dataset(path_to_data=root_path, frac_test=0.2) #20% of the whole dataset is used for testing

    #
    # CLASSIFICATION MODEL
    # path_to_data = 'data/labelledData/fitparams_data'
    path_to_data = 'data/labelledData/labelledData'
    path_to_data = 'data/labelledData/labelledData_gpuSamples_alot/'
    output_path = 'OUTPUT/bdt'
    for d in ['prediction_testdataset', 'TestOnData']:
        try:
            os.mkdir('/'.join([output_path, d]))
        except:
            pass
    classifier = BDT_Classifier(path_to_data=path_to_data, output_path=output_path)
    params = classifier.tune_hyperamaters()
    classifier.train(params=params)
    

    # #
    # REGRESSION MODEL
    # path_to_data = 'data/labelledData/labelledData'
    regressor = BDT_Regressor(path_to_data=path_to_data, output_path='OUTPUT/bdt')
    params = regressor.tune_hyperparameters()
    regressor.train(params=params)
    regressor.classify_predicted_integralMaxdev(path_to_classifier_model='OUTPUT/bdt/classifier_bdt_model.json')

    # # CLASSIFICATION OF THE FIT PARAMETERS : DATA
    combined_classifier = Classify(regressor_model='OUTPUT/bdt/regressor_bdt_model.json',
                                     classifier_model='OUTPUT/bdt/classifier_bdt_model.json',
                                     path_to_data='data/labelledData',
                                     output_path='OUTPUT/bdt/TestOnData')
    combined_classifier.run_classification(info_data_dict={'key_in_name': 'fit_results',
                                                           'file_ext': '.csv',
                                                           'sep': ','},
                                            plot2dcorr=True, plotconfmatrix=True)

    #
    # # TEST PRECLASSIFIER USING WAVEFORM DIRECTLY FROM A ROOT FILE
    test_ = TestPreclassifier(path_to_root_file='raw_waveforms_run_30413.root', hist_prefix='hist_0')
    for chn in range(300):
        test_.run(path_to_model='OUTPUT/bdt/preclassifier/preclassifier.json', chn=chn)