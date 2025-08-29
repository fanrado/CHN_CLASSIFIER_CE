import os, sys
import pandas as pd
import numpy as np
from BDT import split_train_test_dataset, BDT_Classifier, BDT_Regressor, Classify, preClassifier, TestPreclassifier, Sim_waveform, ToyTestPreclassifier

training                    = True
split_dataset_classifier    = False # set to True if you need to split the dataset using the classification stage into train and test
run_classifier              = False
run_regression              = False
generate_npy                = True # set to true if you need to generate the npy files for the preclassification
run_preclassification       = True
run_testpreclassification   = False
# N_samples                 = 3000
all_nsamples              = [10000+i*20000 for i in range(20)]
# all_nsamples              = [200000]
# all_nsamples                = [385000]
if training:
    print("All number of samples to be used: ", all_nsamples)
    print("Press Enter to continue....")
    sys.stdin.read(1)

# N_samples = None
if __name__=='__main__':
    for N_samples in all_nsamples:
        # path_to_data = 'data/labelledData/labelledData_gpuSamples_alot/'
        # output_path = f'OUTPUT/bdt_Ntotsamples_{N_samples}'

        path_to_data = 'DATASET_and_OUTPUT/fine_resolution/data/synthetic_dataset'
        output_path = f'DATASET_and_OUTPUT/fine_resolution/OUTPUT/bdt_Ntotsamples_{N_samples}'
        # path_to_data = 'data/labelledData'
        # output_path = f'OUTPUT/bdt_fitresults'
        try:
            os.mkdir(output_path)
        except:
            pass

        for d in ['prediction_testdataset', 'TestOnData', 'preclassifier']:
            try:
                os.mkdir('/'.join([output_path, d]))
            except:
                pass
        # train_test split
        if split_dataset_classifier:
            # N = 3000
            # root_path = 'data/labelledData/labelledData_gpuSamples_alot'
            root_path = 'DATASET_and_OUTPUT/fine_resolution/data/synthetic_dataset'
            OUT_PATH = '/'.join([root_path, f'Ntotsamples_{N_samples}'])
            # root_path = path_to_data
            # OUT_PATH = root_path
            split_train_test_dataset(path_to_data=root_path, output_path=OUT_PATH, frac_test=0.2, N1=N_samples, N2=2*N_samples, N3=2*N_samples, N4=N_samples) #20% of the whole dataset is used for testing

        #
        path_to_data = f'{path_to_data}/Ntotsamples_{N_samples}'
        try:
            os.mkdir(path_to_data)
        except:
            pass
        # path_to_data = path_to_data
        # CLASSIFICATION MODEL
        if run_classifier:
            classifier = BDT_Classifier(path_to_data=path_to_data, output_path=output_path)
            params = classifier.tune_hyperamaters()
            classifier.train(params=params)
        

        # #
        # REGRESSION MODEL
        if run_regression:
            # path_to_data = 'data/labelledData/labelledData'
            regressor = BDT_Regressor(path_to_data=path_to_data, output_path=output_path)
            params = regressor.tune_hyperparameters()
            regressor.train(params=params)
            regressor.classify_predicted_integralMaxdev(path_to_classifier_model=f'{output_path}/classifier_bdt_model.json')

            # # CLASSIFICATION OF THE FIT PARAMETERS : DATA
            combined_classifier = Classify(regressor_model=f'{output_path}/regressor_bdt_model.json',
                                            classifier_model=f'{output_path}/classifier_bdt_model.json',
                                            path_to_data='DATASET_and_OUTPUT/fine_resolution/data/fit_results_realdata', # path to real data (labelled, for comparison with prediction)
                                            output_path=f'{output_path}/TestOnData')
            combined_classifier.run_classification(info_data_dict={'key_in_name': 'fit_results',
                                                                'file_ext': '.csv',
                                                                'sep': ','},
                                                    plot2dcorr=True, plotconfmatrix=True)
        

        # # #
        # # GENERATING THE npy FILES FOR THE PRECLASSIFICATION
        if generate_npy:
            # path_to_simdata = 'data/labelledData/labelledData_gpuSamples_alot'
            path_to_simdata = 'DATASET_and_OUTPUT/fine_resolution/data/synthetic_dataset'
            OUT_PATH = '/'.join([path_to_simdata, f'Ntotsamples_{N_samples}'])
            # path_to_simdata = path_to_data
            # OUT_PATH = path_to_simdata
            try:
                os.mkdir('/'.join([OUT_PATH, 'npy']))
            except:
                pass
            # class c1
            print('Generating wf for class c1...')
            filename = 'generate_new_samples_c1.csv'
            # sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generatedSamples/generated_new_samples_c1_labelled_tails.csv',
            #                       output_path='data/labelledData/labelledData/WF_sim/')
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            # sim_wf_obj.data2npy()
            sim_wf_obj.data2npy_torch()

            # class c2
            print('Generating wf for class c2...')
            filename = 'generate_new_samples_c2.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            # sim_wf_obj.data2npy()
            sim_wf_obj.data2npy_torch()

            # class c3
            print('Generating wf for class c3...')
            filename = 'generate_new_samples_c3.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            # sim_wf_obj.data2npy()
            sim_wf_obj.data2npy_torch()

            # class c4
            print('Generating wf for class c4...')
            filename = 'generate_new_samples_c4.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            # sim_wf_obj.data2npy()
            sim_wf_obj.data2npy_torch()

        # # PRECLASSIFIER
        if run_preclassification:
            preclassifier_obj = preClassifier(path_to_train=f'{path_to_data}/npy', output_path=output_path)
            preclassifier_obj.run()
            test_ = TestPreclassifier(path_to_root_file='raw_waveforms_run_30413.root', hist_prefix='hist_1', output_path=output_path)
            test_.run(path_to_model=f'{output_path}/preclassifier/preclassifier.json', savefig=True, Nchannels=2000)

    if run_testpreclassification:
        output_path = f'DATASET_and_OUTPUT/fine_resolution/OUTPUT/test_preclassifier'
        # # # TEST PRECLASSIFIER USING WAVEFORM DIRECTLY FROM A ROOT FILE
        # test_ = TestPreclassifier(path_to_root_file='raw_waveforms_run_30413.root', hist_prefix='hist_1', output_path=output_path)
        # test_.run(path_to_model=f'{output_path}/preclassifier.json', savefig=True, Nchannels=1000)
        test_ = ToyTestPreclassifier(peaktype='Positive', rootfilename='magnify-30413-8.root', output_path=output_path)
        test_.run(path_to_model= f'{output_path}/preclassifier.json', Nchannels=100)
