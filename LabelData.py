import pandas as pd
import numpy as np
import os, sys
from response import response, response_legacy
# torch response
import torch, time
from response import response_torch, response_legacy_torch

import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from scipy.stats import gaussian_kde


class LabelData:
    """
    A class for processing and labeling Channel response data.
    This class handles reading source data, calculating response characteristics,
    and classifying Channel responses into different categories based on their
    integral values and maximum deviations from ideal responses.
    """
    def __init__(self, root_path: str, filename=None, fixHeader=False, sep='\t', generate_new_data=False):
        """
            Initialize the LabelData object.
            
            Parameters:
                root_path (str): Path to the directory containing the source data.
                filename (str): Name of the data file to process.
                fixHeader (bool): Flag to fix header issues in the source data if needed.
            In case of generate_new_data = True, filename should be a list of the datasource filenames.
        """
        self.fixHeader = fixHeader
        self.root_path = root_path
        self.filename = filename

        if not generate_new_data:
            print('Current file : ', self.filename)
            self.source_data = self.__read_sourcedata(filename=self.filename, sep=sep)
            self.Fig_output_path = '/'.join([self.root_path, filename.split('.')[0]+'_fig'])
            try:
                os.mkdir(self.Fig_output_path)
            except:
                pass
        else:
            self.source_data = self.__read_all_data(sep=sep)
            self.all_columns = [c for c in self.source_data.columns if 'class' not in c]
        self.response_params = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.data_output_path = '/'.join([self.root_path, 'labelledData'])
        try:
            os.mkdir(self.data_output_path)
        except:
            pass

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select cuda if available

    def __read_sourcedata(self, filename='', sep='\t'):
        """
            Read and preprocess the source data from a TSV file (the data we have are saved in a txt format but have tab as separation of the columns).
            Handles column name cleaning and header fixing if needed.
            
            Returns:
                pandas.DataFrame: Preprocessed source data
        """
        data = pd.read_csv('/'.join([self.root_path, filename]), sep=sep)
        data.columns = data.columns.str.strip().str.replace(' ', '')
        if self.fixHeader:
            columns = data.columns
            data = data.reset_index()
            data.drop(columns[-1], axis=1, inplace=True)
            data.columns = columns
        return data
    
    def local_average_convolve(self, arr, window_size=3):
        """
            Calculate moving average using convolution.
            
            Parameters:
                arr: Input array to be averaged
                window_size (int): Size of the moving average window
                
            Returns:
                numpy.ndarray: Array containing moving averages
        """
        # create weights for averaging
        weights = np.ones(window_size) / window_size
        # calculate moving average
        avg = np.convolve(arr, weights, mode='valid')
        return avg
    
    def find_intersection(self, x1, y1, x2, y2):
        """
            Find intersection of two curves using interpolation
            
            Parameters:
                x1, y1: coordinates of first curve
                x2, y2: coordinates of second curve
            
            Returns:
                x, y: coordinates of intersection point
        """
        # Create interpolation functions for both curves
        f1 = interpolate.interp1d(x1, y1)
        f2 = interpolate.interp1d(x2, y2)
        
        # Find the overlapping x-range
        x_min = max(x1.min(), x2.min())
        x_max = min(x1.max(), x2.max())
        
        # Create array of x-values to search for intersection
        x = np.linspace(x_min, x_max, 10000)
        
        # Calculate y-values for both functions
        y1_interp = f1(x)
        y2_interp = f2(x)
        
        # # Find where the difference is closest to zero
        idx = np.argmin(np.abs(y1_interp - y2_interp))
    
        return x[idx], y1_interp[idx]

    def calculate_Integral_MaxDev(self, tmpdata, returnIdeal=False, plotHist=False):
        """
            Calculate integral values and maximum deviations between real and ideal responses using their tails.
            
            Parameters:
                tmpdata: Input data containing response parameters.
                returnIdeal (bool): If True, return ideal response integrals.
                plotHist (bool): If True, plot histograms of results.
                
            Returns:
                tuple: Contains integral values and maximum deviations.
        """
        integrals_R_selected = []
        integrals_R_ideal_selected = []
        max_deviations = []
        for i in range(len(tmpdata)):
            x = np.linspace(tmpdata['t'].iloc[i], tmpdata['t'].iloc[i]+70, 70)
            par0 = list(tmpdata[self.response_params].iloc[i])
            # try:
            # calculate the response
            R = response(x=x, par=par0)
            R_ideal = response_legacy(x=x, par=par0)
            # find peak in ideal response
            pos_peak = np.argmax(R_ideal)
            # considering the peak time is 2us and each tick corresponds to 0.512 us, there are at most 5 ticks from the peak to the pedestal.
            xtail = x[pos_peak+6:]
            # find intersection
            x1 = x[pos_peak+6:]
            y1 = R[pos_peak+6:]
            x2 = x[pos_peak+6:]
            y2 = R_ideal[pos_peak+6:]
            try:
                x_intersect, y_intersect = self.find_intersection(x1,y1,x2, y2)
            except:
                continue
            # select data between pos_peak+6 and x_intersect
            # mask = x1 <= x_intersect
            mask = x1 <= x[pos_peak+50] # what if not finding the intersection ==> fixing the integration domain
            x_selected = x1[mask]
            R_selected = y1[mask]
            R_ideal_selected = y2[mask]
            integral_R_selected = integrate.simpson(x=x_selected, y=R_selected)
            integral_R_ideal_selected = integrate.simpson(x=x_selected, y=R_ideal_selected)
            integrals_R_selected.append(integral_R_selected)
            integrals_R_ideal_selected.append(integral_R_ideal_selected)
            #
            # Deviation between the ideal and real responses:
            R_avg = self.local_average_convolve(R_selected, 2)
            x_avg = self.local_average_convolve(x_selected, 2)
            R_ideal_avg = self.local_average_convolve(R_ideal_selected, 2)
            deviations = R_avg - R_ideal_avg
            # deviations = R_selected - R_ideal_selected
            max_deviation = np.max(deviations)
            # if np.abs(np.min(deviations)) > np.max(deviations):
            if np.abs(np.min(deviations)) > np.abs(np.max(deviations)):
                max_deviation = np.min(deviations)
            max_deviations.append(max_deviation)
            # time_peaks_diff.append(x[np.argmax(R)] - x[pos_peak])
        integrals_R_selected = np.array(integrals_R_selected)
        integrals_R_ideal_selected = np.array(integrals_R_ideal_selected)
        max_deviations = np.array(max_deviations)
        if plotHist:
            self.__plot_Integral_MaxDev(intIdeal=integrals_R_ideal_selected,
                                        intResp=integrals_R_selected,
                                        MaxDev=max_deviations)
        if returnIdeal:
            return integrals_R_ideal_selected, integrals_R_selected, max_deviations
        else:
            return integrals_R_selected, max_deviations
    
    # def calculate_Integral_MaxDev_gpu(self, x: torch.Tensor, par0: torch.Tensor):
    #     # real and ideal response
    #     R       = response_torch(x=x, par=par0)
    #     R_ideal = response_legacy_torch(x=x, par=par0)

    #     pos_peak = torch.argmax(R_ideal)
    #     x_tail   = x[pos_peak+6:]
    #     R_tail   = R[pos_peak+6:]
    #     I_tail   = R_ideal[pos_peak+6:] # tail of the ideal response

    #     # fixing the integration window to pospeak+50
    #     cutoff   = x_tail[50]
    #     mask     = x_tail <= cutoff

    #     x_sel    = x_tail[mask]
    #     R_sel    = R_tail[mask]
    #     I_sel    = I_tail[mask]

    #     # integrate
    #     int_R    = torch.trapz(R_sel, x_sel)

    #     # 2-point moving average
    #     avg_R    = (R_sel[:-1]  + R_sel[1:]) * 0.5
    #     avg_I    = (I_sel[:-1]  + I_sel[1:]) * 0.5

    #     # max deviation
    #     devs     = avg_R - avg_I
    #     max_p    = torch.max(devs)
    #     max_n    = torch.min(devs)
    #     max_dev  = max_n if torch.abs(max_n) > torch.abs(max_p) else max_p

    #     return int_R, max_dev
        
    def classifyResponse(self, integrals_R, max_deviations, source_data, plotHistClasses=False, plotComparisonResponses=False):
        """
            Classify responses into 4 categories based on integral values and maximum deviations.
            
            Categories:
                - Class 1: negative integral, negative max deviation
                - Class 2: negative integral, positive max deviation
                - Class 3: positive integral, negative max deviation
                - Class 4: positive integral, positive max deviation
            
            Parameters:
                integrals_R: Array of response integral values
                max_deviations: Array of maximum deviations
                plotHistClasses (bool): If True, plot histograms for each class
                plotComparisonResponses (bool): If True, plot comparison of responses
                
            Returns:
                pandas.DataFrame: Classified data with added class labels
        """
        source_data['integral_R'] = integrals_R
        source_data['max_deviation'] = max_deviations
        # cuts before March 22, 2025
        # class1_mask = (source_data['integral_R'] <= 0) & (source_data['max_deviation'] < 0)
        # class2_mask = (source_data['integral_R'] <= 0) & (source_data['max_deviation'] > 0)
        # class3_mask = (source_data['integral_R'] > 0) & (source_data['max_deviation'] < 0)
        # class4_mask = (source_data['integral_R'] > 0) & (source_data['max_deviation'] > 0)

        # FIX THE CONDITIONS DEFINING EACH CLASS
        class1_mask = (source_data['integral_R'] < 0) & (source_data['max_deviation'] < 0) # didn't have <0 before March 22. The <=0 included a conflict with class2
        class2_mask = (source_data['integral_R'] <= 0) & (source_data['max_deviation'] > 0) 
        class3_mask = (source_data['integral_R'] > 0) & (source_data['max_deviation'] <= 0) # didn't have <=0 before March 22
        class4_mask = (source_data['integral_R'] > 0) & (source_data['max_deviation'] > 0)
    
        class1_df = source_data[class1_mask].copy().reset_index().drop('index', axis=1)
        class2_df = source_data[class2_mask].copy().reset_index().drop('index', axis=1)
        class3_df = source_data[class3_mask].copy().reset_index().drop('index', axis=1)
        class4_df = source_data[class4_mask].copy().reset_index().drop('index', axis=1)
        class1_df['class'] = ['c1' for _ in range(len(class1_df))]
        class2_df['class'] = ['c2' for _ in range(len(class2_df))]
        class3_df['class'] = ['c3' for _ in range(len(class3_df))]
        class4_df['class'] = ['c4' for _ in range(len(class4_df))]
        if plotHistClasses:
            self.__plot_HistClasses(class1_df=class1_df, class2_df=class2_df,
                                    class3_df=class3_df, class4_df=class4_df)
        if plotComparisonResponses:
            self.__plot_Comparison_ResponseAndIdeal(class1_df=class1_df, class2_df=class2_df,
                                                  class3_df=class3_df, class4_df=class4_df)
        output_df = pd.concat([class1_df, class2_df, class3_df, class4_df], axis=0)
        output_df = output_df.reset_index().drop('index', axis=1)
        cols_to_drop = [col for col in output_df.columns if 'Unnamed' in col]
        output_df.drop(columns=cols_to_drop, inplace=True)
        output_df['#Ch.#'] = output_df['#Ch.#'].astype('int32').abs()
        return output_df
    
    def __plot_Integral_MaxDev(self, intIdeal, intResp, MaxDev):
        """
            Plot histograms of integral values and maximum deviations.
            
            Parameters:
                intIdeal: Array of ideal response integral values
                intResp: Array of real response integral values
                MaxDev: Array of maximum deviations
        """
        plt.figure()
        plt.hist(intResp, bins=100, histtype='step', label='real response')
        plt.hist(intIdeal, bins=100, histtype='step', label='ideal response')
        plt.xlabel('integral of tail')
        plt.ylabel('count')
        plt.yscale('log')
        # plt.xlim([-2, 10])
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

        plt.figure()
        plt.hist(MaxDev, bins=100, histtype='step', label='max deviation normalized to the maximum')
        plt.xlabel('maximum deviation')
        plt.ylabel('count')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.close()

    def __plot_HistClasses(self, class1_df, class2_df, class3_df, class4_df):
        """
            Plot histograms of maximum deviations for each response class.
            
            Parameters:
                class1_df: DataFrame containing Class 1 responses
                class2_df: DataFrame containing Class 2 responses
                class3_df: DataFrame containing Class 3 responses
                class4_df: DataFrame containing Class 4 responses
        """
        plt.figure()
        class1_df['max_deviation'].hist(bins=100, histtype='step', label='class_1')
        class2_df['max_deviation'].hist(bins=100, histtype='step', label='class_2')
        class3_df['max_deviation'].hist(bins=100, histtype='step', label='class_3')
        class4_df['max_deviation'].hist(bins=100, histtype='step', label='class_4')
        plt.xlabel('max deviation')
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.close()

    def __plot_Comparison_ResponseAndIdeal(self, class1_df, class2_df, class3_df, class4_df):
        """
            Plot comparison between real and ideal responses for each class.
            
            Parameters:
                class1_df: DataFrame containing Class 1 responses
                class2_df: DataFrame containing Class 2 responses
                class3_df: DataFrame containing Class 3 responses
                class4_df: DataFrame containing Class 4 responses
        """
        # class1
        x = np.linspace(class1_df.copy().reset_index()['t'].iloc[2], class1_df.copy().reset_index()['t'].iloc[2]+70, 70)
        par0 = list(class1_df.copy().reset_index()[self.response_params].iloc[2])
        R = response(x=x, par=par0)
        R_ideal = response_legacy(x=x, par=par0)

        plt.figure()
        plt.plot(x, R, label='real response class1')
        plt.plot(x, R_ideal, label='ideal')
        plt.xlabel('Ticks (0.512 us/Tick)')
        plt.legend()
        plt.show()
        plt.close()

        # class2
        x = np.linspace(class2_df.copy().reset_index()['t'].iloc[2], class2_df.copy().reset_index()['t'].iloc[2]+70, 70)
        par0 = list(class2_df.copy().reset_index()[self.response_params].iloc[2])
        R = response(x=x, par=par0)
        R_ideal = response_legacy(x=x, par=par0)

        plt.figure()
        plt.plot(x, R, label='real response class2')
        plt.plot(x, R_ideal, label='ideal')
        plt.xlabel('Ticks (0.512 us Tick)')
        plt.legend()
        plt.show()
        plt.close()

        # class3
        x = np.linspace(class3_df.copy().reset_index()['t'].iloc[2], class3_df.copy().reset_index()['t'].iloc[2]+70, 70)
        par0 = list(class3_df.copy().reset_index()[self.response_params].iloc[2])
        R = response(x=x, par=par0)
        R_ideal = response_legacy(x=x, par=par0)

        plt.figure()
        plt.plot(x, R, label='real response class3')
        plt.plot(x, R_ideal, label='ideal')
        plt.xlabel('Ticks (0.512 us Tick)')
        plt.legend()
        plt.show()
        plt.close()

        # class4
        x = np.linspace(class4_df.copy().reset_index()['t'].iloc[2], class4_df.copy().reset_index()['t'].iloc[2]+70, 70)
        par0 = list(class4_df.copy().reset_index()[self.response_params].iloc[2])
        R = response(x=x, par=par0)
        R_ideal = response_legacy(x=x, par=par0)

        plt.figure()
        plt.plot(x, R, label='real response class4')
        plt.plot(x, R_ideal, label='ideal')
        plt.xlabel('Ticks (0.512 us Tick)')
        plt.legend()
        plt.show()
        plt.close()
    
    def runLabelling(self):
        """
            Execute the complete labelling process:
            1. Calculate integral values and maximum deviations
            2. Classify responses
            3. Save results to CSV file
            
            The output file is saved in the data_output_path directory with '_labelled_tails.csv' suffix.
        """
        integrals_R_selected, max_deviations = self.calculate_Integral_MaxDev(tmpdata=self.source_data, returnIdeal=False,
                                                                              plotHist=False)
        labelledData = self.classifyResponse(integrals_R=integrals_R_selected, max_deviations=max_deviations, source_data=self.source_data, plotHistClasses=False,
                                             plotComparisonResponses=False)
        outputfilename = self.filename.split('.')[0] + '_labelled_tails.csv'
        labelledData.to_csv('/'.join([self.data_output_path, outputfilename]))

    def __read_all_data(self, sep='\t'):
        data = pd.DataFrame()
        for i, f in enumerate(self.filename): # self.filename here should be a list but I don't want to change the name of the variable because the class also deal with single file
            tmpdata = self.__read_sourcedata(filename=f, sep=sep)
            if i==0:
                data = tmpdata.copy()
            else:
                data = pd.concat([data, tmpdata], axis=0)

        return data
            
    def GenerateNewSamples(self, N_samples=1000, target_class='c2'):
        output_filename = 'generated_new_samples'
        # class c2 and c3 were chosen because they don't have many sample points.
        # The parameters we can generate from these two classes are not 100% sure to be class c2 or c3. They can always generate c1 and c4.
        # If c1 and c4 are not enough, we can generate more of them later.
        if target_class=='c2':
            data = self.source_data[self.all_columns][self.source_data['class']=='c2'].to_numpy()
        elif isinstance(target_class, list):
            data = self.source_data[self.all_columns][(self.source_data['class']=='c3') | (self.source_data['class']=='c2') | (self.source_data['class']=='c1')| (self.source_data['class']=='c4')].to_numpy()
            target_class = target_class[0]
        data_T = data.T
        kde = gaussian_kde(data_T, bw_method='scott')

        c_df = pd.DataFrame()
        Total_Number_c = 0

        while Total_Number_c < N_samples:
            new_samples = kde.resample(1)
            new_samples = new_samples.T
            new_df = pd.DataFrame(new_samples, columns=self.all_columns)
            integrals_R_selected, max_deviations = self.calculate_Integral_MaxDev(tmpdata=new_df, returnIdeal=False,
                                                                                plotHist=False)
            
            if (np.abs(integrals_R_selected[0]) > 10000) or (np.abs(max_deviations[0]) > 1000):
                continue
            labelledData = self.classifyResponse(integrals_R=integrals_R_selected, max_deviations=max_deviations, source_data=new_df,
                                                 plotHistClasses=False, plotComparisonResponses=False)
            
            if (Total_Number_c < N_samples) and (labelledData['class'].iloc[0]==target_class):
                if Total_Number_c==0:
                    c_df = labelledData
                    Total_Number_c += 1
                else:
                    c_df = pd.concat([c_df, labelledData], axis=0)
                    Total_Number_c += 1
            if Total_Number_c == N_samples:
                outputfilename = output_filename + f'_{target_class}_labelled_tails.csv'
                c_df.to_csv('/'.join([self.data_output_path, outputfilename]))
    
    # def GenerateNewSamples_gpu(self, N_samples=1000, target_class='c2'):
    #     output_filename = 'generate_new_samples'
    #     # class c2 and c3 were chosen because they don't have many sample points.
    #     # The parameters we can generate from these two classes are not 100% sure to be class c2 or c3. They can always generate c1 and c4.
    #     # If c1 and c4 are not enough, we can generate more of them later.
    #     if target_class=='c2':
    #         data = self.source_data[self.all_columns][self.source_data['class']=='c2'].to_numpy()
    #     else:
    #         data = self.source_data[self.all_columns][(self.source_data['class']=='c3') | (self.source_data['class']=='c2') | (self.source_data['class']=='c1')| (self.source_data['class']=='c4')].to_numpy()
        
    #     data_T = data.T
    #     kde = gaussian_kde(data_T, bw_method='scott')

    #     c_df = pd.DataFrame()
    #     Total_Number_c = 0
    #     while Total_Number_c < N_samples:
    #         new_samples = kde.resample(1)
    #         new_samples = new_samples.T
    #         new_df = pd.DataFrame(new_samples, columns=self.all_columns)
    #         #
    #         # GPU-accelerated integral of the tail and max deviation
    #         x = torch.linspace(
    #             float(new_df['t'].iloc[0]),
    #             float(new_df['t'].iloc[0]) + 70,
    #             steps=70,
    #             dtype=torch.float32,
    #             device=self.device
    #         )
    #         par0 = torch.tensor(
    #             new_df[self.response_params].iloc[0].values,
    #             dtype=torch.float32,
    #             device=self.device
    #         )
    #         int_R, max_dev = self.calculate_Integral_MaxDev_gpu(x=x, par0=par0)
    #         integrals_R_selected = np.array([int_R.item()])
    #         max_deviations       = np.array([max_dev.item()])
            
    #         if (np.abs(integrals_R_selected[0]) > 10000) or (np.abs(max_deviations[0]) > 1000):
    #             continue
    #         labelledData = self.classifyResponse(integrals_R=integrals_R_selected, max_deviations=max_deviations, source_data=new_df,
    #                                              plotHistClasses=False, plotComparisonResponses=False)
            
    #         if (Total_Number_c < N_samples) and (labelledData['class'].iloc[0]==target_class):
    #             if Total_Number_c==0:
    #                 c_df = labelledData
    #                 Total_Number_c += 1
    #             else:
    #                 c_df = pd.concat([c_df, labelledData], axis=0)
    #                 Total_Number_c += 1
            
    #         if Total_Number_c == N_samples:
    #             outputfilename = output_filename + f'_{target_class}_labelled_tails.csv'
    #             c_df.to_csv('/'.join([self.data_output_path, outputfilename]))

    def calculate_Integral_MaxDev_gpu_batch(self, pars: torch.Tensor):
        """
        pars: [B,7] on self.device
        returns (ints, max_devs) as numpy arrays of shape [B]
        """
        B, D = pars.shape
        L = 70
        # build time‐grid: [B,L]
        t0 = pars[:,0].unsqueeze(1)   # [B,1]
        xs = t0 + torch.arange(L, device=self.device).view(1, L)

        # compute in batch
        R      = response_torch(xs, pars)              # [B,L]
        R_ideal= response_legacy_torch(xs, pars)       # [B,L]

        # find peak
        pos_peak = torch.argmax(R_ideal, dim=1)        # [B]

        # gather fixed‐length tail 50
        idx_base = pos_peak.unsqueeze(1) + torch.arange(50, device=self.device).view(1,50)
        R_tail       = R     .gather(1, idx_base)      # [B,50]
        R_ideal_tail = R_ideal.gather(1, idx_base)     # [B,50]

        # integral via trapezoid
        ints = ((R_tail[:,:-1] + R_tail[:,1:])*0.5).sum(dim=1)  # [B]

        # 2‐pt avg via conv1d
        kern = torch.tensor([0.5,0.5], device=self.device).view(1,1,2)
        R_avg       = torch.nn.functional.conv1d(R_tail.unsqueeze(1), kern).squeeze(1)      # [B,49]
        R_ideal_avg = torch.nn.functional.conv1d(R_ideal_tail.unsqueeze(1), kern).squeeze(1) # [B,49]

        # max deviation (signed)
        dev = R_avg - R_ideal_avg                                          # [B,49]
        mn, _ = dev.min(dim=1);  mx, _ = dev.max(dim=1)
        max_dev = torch.where(mn.abs() > mx.abs(), mn, mx)                # [B]

        return ints.cpu().numpy(), max_dev.cpu().numpy()


    def GenerateNewSamples_gpu(self, N_samples=1000, target_class='c2', batch_size=256):
        # … your setup up to kde …
        if target_class=='c2':
            data = self.source_data[self.all_columns][self.source_data['class']=='c2'].to_numpy()
        else:
            data = self.source_data[self.all_columns].to_numpy()
        kde = gaussian_kde(data.T, bw_method='scott')

        out_rows = []
        while len(out_rows) < N_samples:
            # 1) draw batch_size new params via KDE
            new_params = kde.resample(batch_size).T                             # [B,D] CPU
            pars_t = torch.tensor(new_params, dtype=torch.float32, device=self.device)

            # 2) compute ints & devs in one GPU call
            ints, devs = self.calculate_Integral_MaxDev_gpu_batch(pars_t)       # each shape [B]

            # 3) filter too‐large tails
            mask1 = (np.abs(ints) <= 1e4) & (np.abs(devs) <= 1e3)

            if not mask1.any():
                continue

            # 4) classify these masked ones on CPU *without* looping in python
            df = pd.DataFrame(new_params[mask1], columns=self.all_columns)
            df['integral_R']     = ints[mask1]
            df['max_deviation']  = devs[mask1]

            # vectorized class assignment:
            df['class'] = np.where(
               (df['integral_R']<0)&(df['max_deviation']<0), 'c1',
            np.where((df['integral_R']<=0)&(df['max_deviation']>0), 'c2',
            np.where((df['integral_R']>0)&(df['max_deviation']<=0), 'c3','c4')))

            # 5) select only target_class
            sel = df['class']==target_class
            if sel.any():
                out_rows.append(df[sel])
                # flatten list-of-frames if we overshot:
                all_df = pd.concat(out_rows, axis=0).head(N_samples)
        
        # drop unnecessary column
        cols_to_drop = [col for col in all_df.columns if 'Unnamed' in col]
        all_df.drop(columns=cols_to_drop, inplace=True)
        all_df['#Ch.#'] = all_df['#Ch.#'].astype('int32').abs()
        # save final
        all_df.to_csv(f"{self.data_output_path}/generate_new_samples_{target_class}.csv", index=False)

if __name__ == '__main__':
    ## LABELLING THE DATA
    # labeldata_obj = LabelData(root_path='data/', filename='fit_results_run_30404_no_avg.txt', fixHeader=False)
    # labeldata_obj.runLabelling()
    # list_file_source = [f for f in os.listdir('data/fit_params/Fit_Results') if '.txt' in f]
    # list_file_source = [f for f in os.listdir('data/kde_syntheticdata')]
    # list_file_source = [f for f in os.listdir('data') if '.csv' in f]
    # for f in list_file_source:
    #     labeldata_obj = LabelData(root_path='data//fit_params/Fit_Results', filename=f, fixHeader=False, sep='\t')
    #     labeldata_obj.runLabelling()
    ##
    # ## GENERATE NEW DATASET
    # list_file_source = [f for f in os.listdir('data/labelledData') if ('.csv' in f) and ('kde' not in f)]
    # labeldata_obj = LabelData(root_path='data/labelledData', filename=list_file_source, fixHeader=False, sep=',', generate_new_data=True)
    # labeldata_obj.GenerateNewSamples(N_samples=1000, target_class=['c1', 'c3', 'c4'])
    # labeldata_obj.GenerateNewSamples(N_samples=200000, target_class=['c1'])
    # labeldata_obj.GenerateNewSamples(N_samples=200000, target_class=['c4'])
    # labeldata_obj.GenerateNewSamples(N_samples=100000, target_class=['c3'])
    # labeldata_obj.GenerateNewSamples(N_samples=100000, target_class='c2')

    ## GENERATE NEW DATASET using GPU
    # related to GPU kernel time
    start_evt   = torch.cuda.Event(enable_timing=True)
    end_evt     = torch.cuda.Event(enable_timing=True)
    # start is for the total time CPU + GPU
    start = time.perf_counter()
    torch.cuda.synchronize()    # drain any prior work
    list_file_source = [f for f in os.listdir('data/labelledData') if ('.csv' in f) and ('kde' not in f)]
    labeldata_obj = LabelData(root_path='data/labelledData', filename=list_file_source, fixHeader=False, sep=',', generate_new_data=True)
    print('Class c1')
    start_evt.record()
    # labeldata_obj.GenerateNewSamples_gpu(N_samples=1000, target_class='c1')
    labeldata_obj.GenerateNewSamples_gpu(N_samples=300000, target_class='c1', batch_size=32)
    end_evt.record()
    
    torch.cuda.synchronize()    # wait until all GPU operations are done
    total = time.perf_counter() - start
    print(f'Total elpsed time : {total:.3f} s')
    print(f'GPU kernel time : {start_evt.elapsed_time(end_evt):.1f} ms')
    # c4
    # print('Class c4')
    # start_evt.record()
    # # labeldata_obj.GenerateNewSamples_gpu(N_samples=1000, target_class='c1')
    # labeldata_obj.GenerateNewSamples_gpu(N_samples=300000, target_class='c4', batch_size=32)
    # end_evt.record()
    # torch.cuda.synchronize()    # wait until all GPU operations are done
    # print(f'GPU kernel time : {start_evt.elapsed_time(end_evt):.1f} ms')
    # #
    # # c3
    # print('Class c3')
    # start_evt.record()
    # # labeldata_obj.GenerateNewSamples_gpu(N_samples=1000, target_class='c1')
    # labeldata_obj.GenerateNewSamples_gpu(N_samples=200000, target_class='c3', batch_size=32)
    # end_evt.record()
    # torch.cuda.synchronize()    # wait until all GPU operations are done
    # print(f'GPU kernel time : {start_evt.elapsed_time(end_evt):.1f} ms')
    # # c2
    # print('Class c2')
    # start_evt.record()
    # # labeldata_obj.GenerateNewSamples_gpu(N_samples=1000, target_class='c1')
    # labeldata_obj.GenerateNewSamples_gpu(N_samples=200000, target_class='c2', batch_size=32)
    # end_evt.record()
    # torch.cuda.synchronize()    # wait until all GPU operations are done
    # print(f'GPU kernel time : {start_evt.elapsed_time(end_evt):.1f} ms')