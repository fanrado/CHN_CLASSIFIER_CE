import os, sys
import pandas as pd
import numpy as np
from BDT import split_train_test_dataset, BDT_Classifier, BDT_Regressor, Classify, preClassifier, TestPreclassifier, Sim_waveform

split_dataset_classifier = False # set to True if you need to split the dataset using the classification stage into train and test
run_classifier = True
run_regression = False
generate_npy = False # set to true if you need to generate the npy files for the preclassification
run_preclassification = True
# N_samples = 3000
# all_nsamples = [3000+i*4000 for i in range(20)]
all_nsamples = [200000]
print("All number of samples to be used: ", all_nsamples)
print("Press Enter to continue....")
sys.stdin.read(1)

# N_samples = None
if __name__=='__main__':
    for N_samples in all_nsamples:
        path_to_data = 'data/labelledData/labelledData_gpuSamples_alot/'
        output_path = f'OUTPUT/bdt_Ntotsamples_{N_samples}'
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
            root_path = 'data/labelledData/labelledData_gpuSamples_alot'
            OUT_PATH = '/'.join([root_path, f'Ntotsamples_{N_samples}'])
            # root_path = path_to_data
            # OUT_PATH = root_path
            split_train_test_dataset(path_to_data=root_path, output_path=OUT_PATH, frac_test=0.2, N1=N_samples, N2=N_samples, N3=N_samples, N4=N_samples) #20% of the whole dataset is used for testing

        #
        path_to_data = f'{path_to_data}/Ntotsamples_{N_samples}'
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
                                            path_to_data='data/labelledData',
                                            output_path=f'{output_path}/TestOnData')
            combined_classifier.run_classification(info_data_dict={'key_in_name': 'fit_results',
                                                                'file_ext': '.csv',
                                                                'sep': ','},
                                                    plot2dcorr=True, plotconfmatrix=True)
        

        # # #
        # # GENERATING THE npy FILES FOR THE PRECLASSIFICATION
        if generate_npy:
            path_to_simdata = 'data/labelledData/labelledData_gpuSamples_alot'
            OUT_PATH = '/'.join([root_path, f'Ntotsamples_{N_samples}'])
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
            sim_wf_obj.data2npy()

            # class c2
            print('Generating wf for class c2...')
            filename = 'generate_new_samples_c2.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            sim_wf_obj.data2npy()

            # class c3
            print('Generating wf for class c3...')
            filename = 'generate_new_samples_c3.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            sim_wf_obj.data2npy()

            # class c4
            print('Generating wf for class c4...')
            filename = 'generate_new_samples_c4.csv'
            sim_wf_obj = Sim_waveform(path_to_sim=f'{path_to_simdata}/{filename}',
                                output_path=f'{OUT_PATH}/npy/', N_samples=N_samples)
            sim_wf_obj.data2npy()

        # # PRECLASSIFIER
        if run_preclassification:
            preclassifier_obj = preClassifier(path_to_train=f'{path_to_data}/npy', output_path=output_path)
            preclassifier_obj.run()

            
            # # TEST PRECLASSIFIER USING WAVEFORM DIRECTLY FROM A ROOT FILE
            test_ = TestPreclassifier(path_to_root_file='raw_waveforms_run_30413.root', hist_prefix='hist_0', output_path=output_path)
            for chn in range(2000):
                test_.run(path_to_model=f'{output_path}/preclassifier/preclassifier.json', chn=chn)