import pandas as pd
import numpy as np
import os, sys
from BDT import ToyTestPreclassifier

'''
    Inputs needed:
        - the preclassifier model : preclassifier.json
        - the data : magnify-{runno}-8.root
        - csv file, results of the preclassification
        - the output directory : DATASET_and_OUTPUT/fine_resolution/OUTPUT/test_preclassifier
        - the fit results of previous runs, saved in DATASET_and_OUTPUT/Fit_Results_realdata/labelledData
'''
runno = sys.argv[1]

if __name__=='__main__':
    # run the preclassification
    output_path = f'DATASET_and_OUTPUT/fine_resolution/OUTPUT/test_preclassifier'
    test_ = ToyTestPreclassifier(peaktype='Positive', rootfilename=f'magnify-{runno}-8.root', output_path=output_path)
    test_.run(path_to_model= f'{output_path}/preclassifier.json')

    #
    path_to_preclass_csv = 'DATASET_and_OUTPUT/fine_resolution/OUTPUT/test_preclassifier'
    pre_class = pd.read_csv('/'.join([path_to_preclass_csv, 'preclassification_ROOT_run_30413.csv']))

    path_to_data = 'DATASET_and_OUTPUT/Fit_Results_realdata/labelledData'
    list_files = os.listdir(path_to_data)
    
    all_df = pd.DataFrame()
    for i, f in enumerate(list_files):
        df = pd.read_csv('/'.join([path_to_data, f]))
        if i==0:
            all_df = df.copy()
        else:
            all_df = pd.concat([all_df, df], ignore_index=True)

    c1_c2 = all_df[all_df['class'].isin(['c1', 'c2'])].copy().reset_index().drop('index', axis=1)
    c3_c4 = all_df[all_df['class'].isin(['c3', 'c4'])].copy().reset_index().drop('index', axis=1)
    
    c1_c2_k3_median = np.round(np.median(c1_c2['k3']), 4)
    c1_c2_k4_median = np.round(np.median(c1_c2['k4']), 4)
    c1_c2_k5_median = np.round(np.median(c1_c2['k5']), 4)
    c1_c2_k6_median = np.round(np.median(c1_c2['k6']), 4)

    c3_c4_k3_median = np.round(np.median(c3_c4['k3']), 4)
    c3_c4_k4_median = np.round(np.median(c3_c4['k4']), 4)
    c3_c4_k5_median = np.round(np.median(c3_c4['k5']), 4)
    c3_c4_k6_median = np.round(np.median(c3_c4['k6']), 4)

    k3_list = []
    k4_list = []
    k5_list = []
    k6_list = []

    for c in pre_class['class']:
        if c=='c1' or c=='c2':
            k3_list.append(c1_c2_k3_median)
            k4_list.append(c1_c2_k4_median)
            k5_list.append(c1_c2_k5_median)
            k6_list.append(c1_c2_k6_median)
        elif c=='c3' or c=='c4':
            k3_list.append(c3_c4_k3_median)
            k4_list.append(c3_c4_k4_median)
            k5_list.append(c3_c4_k5_median)
            k6_list.append(c3_c4_k6_median)
        
    pre_class['k3'] = k3_list
    pre_class['k4'] = k4_list
    pre_class['k5'] = k5_list
    pre_class['k6'] = k6_list

    pre_class.to_csv(f'{path_to_preclass_csv}/preclassified_chn_run_30413.csv', index=False)