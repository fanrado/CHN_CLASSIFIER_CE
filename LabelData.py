import pandas as pd
import numpy as np
import os
from response import response, response_legacy
import matplotlib.pyplot as plt
from scipy import integrate, interpolate

class LabelData:
    """
    A class for processing and labeling Channel response data.
    This class handles reading source data, calculating response characteristics,
    and classifying Channel responses into different categories based on their
    integral values and maximum deviations from ideal responses.
    """
    def __init__(self, root_path: str, filename: str, fixHeader=False, sep='\t'):
        """
            Initialize the LabelData object.
            
            Parameters:
                root_path (str): Path to the directory containing the source data.
                filename (str): Name of the data file to process.
                fixHeader (bool): Flag to fix header issues in the source data if needed.
        """
        self.fixHeader = fixHeader
        self.root_path = root_path
        self.filename = filename
        print('Current file : ', self.filename)
        self.source_data = self.__read_sourcedata(sep=sep)
        self.response_params = ['t', 'A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.Fig_output_path = '/'.join([self.root_path, filename.split('.')[0]+'_fig'])
        self.data_output_path = '/'.join([self.root_path, 'labelledData'])
        try:
            os.mkdir(self.Fig_output_path)
        except:
            pass
        try:
            os.mkdir(self.data_output_path)
        except:
            pass

    def __read_sourcedata(self, sep='\t'):
        """
            Read and preprocess the source data from a TSV file (the data we have are saved in a txt format but have tab as separation of the columns).
            Handles column name cleaning and header fixing if needed.
            
            Returns:
                pandas.DataFrame: Preprocessed source data
        """
        data = pd.read_csv('/'.join([self.root_path, self.filename]), sep=sep)
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
        
    def classifyResponse(self, integrals_R, max_deviations, plotHistClasses=False, plotComparisonResponses=False):
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
        self.source_data['integral_R'] = integrals_R
        self.source_data['max_deviation'] = max_deviations
        # cuts before March 22, 2025
        # class1_mask = (self.source_data['integral_R'] <= 0) & (self.source_data['max_deviation'] < 0)
        # class2_mask = (self.source_data['integral_R'] <= 0) & (self.source_data['max_deviation'] > 0)
        # class3_mask = (self.source_data['integral_R'] > 0) & (self.source_data['max_deviation'] < 0)
        # class4_mask = (self.source_data['integral_R'] > 0) & (self.source_data['max_deviation'] > 0)

        # FIX THE CONDITIONS DEFINING EACH CLASS
        class1_mask = (self.source_data['integral_R'] < 0) & (self.source_data['max_deviation'] < 0) # didn't have <0 before March 22. The <=0 included a conflict with class2
        class2_mask = (self.source_data['integral_R'] <= 0) & (self.source_data['max_deviation'] > 0) 
        class3_mask = (self.source_data['integral_R'] > 0) & (self.source_data['max_deviation'] <= 0) # didn't have <=0 before March 22
        class4_mask = (self.source_data['integral_R'] > 0) & (self.source_data['max_deviation'] > 0)
    
        class1_df = self.source_data[class1_mask].copy().reset_index().drop('index', axis=1)
        class2_df = self.source_data[class2_mask].copy().reset_index().drop('index', axis=1)
        class3_df = self.source_data[class3_mask].copy().reset_index().drop('index', axis=1)
        class4_df = self.source_data[class4_mask].copy().reset_index().drop('index', axis=1)
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
        plt.xlabel('Ticks (0.512$\mu$s/Tick)')
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
        plt.xlabel('Ticks (0.512$\mu$s/Tick)')
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
        plt.xlabel('Ticks (0.512$\mu$s/Tick)')
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
        plt.xlabel('Ticks (0.512$\mu$s/Tick)')
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
        labelledData = self.classifyResponse(integrals_R=integrals_R_selected, max_deviations=max_deviations, plotHistClasses=False,
                                             plotComparisonResponses=False)
        outputfilename = self.filename.split('.')[0] + '_labelled_tails.csv'
        labelledData.to_csv('/'.join([self.data_output_path, outputfilename]))

if __name__ == '__main__':
    ## LABELLING THE DATA
    # labeldata_obj = LabelData(root_path='data/', filename='fit_results_run_30404_no_avg.txt', fixHeader=False)
    # labeldata_obj.runLabelling()
    # list_file_source = [f for f in os.listdir('data/fit_params/Fit_Results')]
    list_file_source = [f for f in os.listdir('data/kde_syntheticdata')]
    # list_file_source = [f for f in os.listdir('data') if '.csv' in f]
    for f in list_file_source:
        labeldata_obj = LabelData(root_path='data/kde_syntheticdata', filename=f, fixHeader=False, sep=',')
        labeldata_obj.runLabelling()
    ##
    ## RANDOMLY GENERATE THE FIT PARAMETERS
    # rndm_fitparams_obj = RandomFitParameters(path_to_data_model='data/labelledData_after_March22_2025', dataFile_name='fit_results_run_30413_no_avg_labelled_tails.csv', output_path='OUTPUT')
    # rndm_fitparams_obj.runAna(plotDist=False)