# Import libraries
import os, sys
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from response import *

import xgboost as xgb
from util_bdt import dataframe2DMatrix, one_hot_encode_sklearn

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
        # N_samples = self.sim_data.shape[0]
        N_samples = 10000
        key_name = self.path_to_sim.split('/')[-1].split('.')[0].split('_')[3]
        for isample in range(N_samples):
            params = list(self.sim_data[self.response_params].iloc[isample])
            R = self.__generate_1wf(params=params)
            dict_params = dict(zip(self.response_params, params))
            dict_params['class'] = self.sim_data['class'].iloc[isample]
            dict_params['wf'] = R
            dict_params['integral_R'] = self.sim_data['integral_R'].iloc[isample]
            dict_params['max_deviation'] = self.sim_data['max_deviation'].iloc[isample]
            np.savez(f'{self.output_path}/wf_{key_name}_{isample}.npz', **dict_params)


class Load_chunk_dset:
    '''
        This class will load a chunk of the data.
    '''
    def __init__(self, path_to_dset: str, chunk_size=5, target_columns=['']):
        self.path_to_dset = path_to_dset
        self.list_dset = ['/'.join([self.path_to_dset, f]) for f in os.listdir(self.path_to_dset)[:5000]]
        self.target_columns = target_columns
        self.input_columns = [f'p{i}' for i in range(70)]
        self.chunk_size = chunk_size
        self.iter = 0

    def npz2df(self, filepath='', forTest_regressor=False):
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
            # return xgb.DMatrix(X, label=None), y
            return dataframe2DMatrix(X=X), y
        else:
            return wf_df

    def reset(self):
        self.iter = 0
        return self
    
    def load(self, tasktype='regression'):
        try:
            chunk = pd.DataFrame()
            if self.chunk_size*(self.iter+1) > len(self.list_dset)-1:
                return None
            for i, ichunk in enumerate(range(self.iter*self.chunk_size, (self.iter+1)*self.chunk_size)):
                df = self.npz2df(filepath=self.list_dset[ichunk])
                if i==0:
                    chunk = df.copy()
                else:
                    chunk = pd.concat([chunk, df], axis=0)
            if len(chunk)==0:
                return None
            self.iter += 1
            if tasktype=='regression':
                # split features (input) and target values
                # y = chunk[self.target_columns].values
                # X = chunk[self.input_columns].values
                # return xgb.DMatrix(X, label=y)
                #
                # use dataframes
                y = chunk[self.target_columns]
                X = chunk[self.input_columns]
                return dataframe2DMatrix(X=X, y=y)
            elif tasktype=='classification':
                tmp_chunk = one_hot_encode_sklearn(data=chunk, column_name='class')
                X = tmp_chunk[self.input_columns]
                y = tmp_chunk[self.target_columns]
                return dataframe2DMatrix(X=X, y=y)
        except:
            return None
    
        
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
    def __init__(self, path_to_data=None, output_path=None, target_columns=['']):
        '''
            path_to_data : path to the list of npz data,
            output_path : path to where you want to save the output of the code.
        '''
        self.path_to_data = path_to_data
        self.output_path = output_path
        self.target_columns = target_columns


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
            'max_depth': 15,
            'learning_rate': 0.4,
            'min_child_weight' : 15,
            'num_boost_round': 200,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }

        data_iter = Load_chunk_dset(path_to_dset=self.path_to_data, chunk_size=200, target_columns=self.target_columns)
        next_chunk = data_iter.load(tasktype=tasktype)
        eval_chunk = data_iter.load(tasktype=tasktype)
        # regressor_model = xgb.XGBRegressor(**params)
        # regressor_model.fit(X=next_chunk[0],y=next_chunk[1])
        bdt_model = xgb.train(params=params,
                                    dtrain = next_chunk,
                                    evals=[(next_chunk, 'train'), (eval_chunk, 'eval')],
                                    early_stopping_rounds=20,
                                    xgb_model=None,
                                    verbose_eval=True)
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
        return bdt_model
    
    def testRegressor(self, regressor_predFitParams=None, regressor_predIntegral=None, regressor_predMaxdev=None, Ntest=100):
        if regressor_predFitParams is None:
            return None
        
        if regressor_predIntegral is None:
            return None
        
        if regressor_predMaxdev is None:
            return None
        
        list_dset = ['/'.join([self.path_to_data, f]) for f in os.listdir(self.path_to_data)[5000:5000+Ntest]]
        target_columns = self.target_columns + ['integral_R', 'max_deviation']
        data_iter = Load_chunk_dset(path_to_dset=self.path_to_data, chunk_size=0, target_columns=target_columns)
        
        # load the regression model trained to predict the value of the integral
        regressor_predIntegral_model = xgb.Booster()
        regressor_predIntegral_model.load_model(regressor_predIntegral)

        # load the regression model trained to predict the value of the maximum deviation between tails
        regressor_predMaxdev_model = xgb.Booster()
        regressor_predMaxdev_model.load_model(regressor_predMaxdev)

        comparison_df = pd.DataFrame()
        for j, f in enumerate(list_dset):
            dtest, ytest = data_iter.npz2df(filepath=f, forTest_regressor=True)
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

        return comparison_df
        
    def classification(self):
        pass

def compare_truth_pred(test_df: pd.DataFrame, output_path: str):
    param = 'integral_R'
    plt.figure()
    plt.hist(test_df[f'{param}_truth_pred'], histtype='step', bins=100, label='truth pred')
    plt.hist(test_df[f'{param}_pred_pred'], histtype='step', bins=100, label='pred pred')
    plt.hist(test_df[f'{param}_truth_truth'], histtype='step', bins=100, label='truth truth')
    plt.title(param)
    # plt.xlim([-2000, 2000])
    plt.xscale('log')
    plt.yscale('log')
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
    plt.xlim([-250, 250])
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{output_path}/comparison_max_deviation.png')
    plt.close()
    
if __name__ == '__main__':
    # Generating training dataset
    sim_wf_obj = Sim_waveform(path_to_sim='data/labelledData/labelledData/generated_new_samples_c3_labelled_tails.csv',
                          output_path='data/labelledData/labelledData/WF_sim/')
    sim_wf_obj.run()

    # Training the regression models to predict the maximum deviation and integral of the tails
    target_columns = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
    # target_columns = ['class_c3']
    # chunk_dset_obj = Load_chunk_dset(path_to_dset='data/labelledData/labelledData/WF_sim', chunk_size=5, target_columns=taget_columns)
    # chunk_dset_obj.test()
    preclassifier_obj = PreClassifier_BDT(path_to_data='data/labelledData/labelledData/WF_sim', output_path='OUTPUT/Preclassifier', target_columns=target_columns)
    regressor_model = preclassifier_obj.Train_bdt(tasktype='regression')
    #
    # Test the regression model and compare the result with the truth
    test_df = preclassifier_obj.testRegressor(regressor_predFitParams=regressor_model, Ntest=5000, regressor_predIntegral='OUTPUT/Kept_RESULTS/OK_SIMULATION_moreSamplesThanApr12_2025/integral_R_model.json',
                                            regressor_predMaxdev='OUTPUT/Kept_RESULTS/OK_SIMULATION_moreSamplesThanApr12_2025/max_deviation_model.json')