import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import uproot

class CSVModel:
    def __init__(self, csv_file_path, regressor_model_file_path, classifier_model_file_path):
        """
        Initialize the CSVModel with the path to the CSV file and the XGBoost model.
        """
        self.csv_file_path = csv_file_path
        self.regressor_model_file_path = regressor_model_file_path
        self.classifier_model_file_path = classifier_model_file_path
        self.data = self.load_csv()
        # print(self.data[self.data['#Ch.#']==5])
        # self.model = self.load_model()
        self.load_model()
        self.input_columns = ['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
        self.class_map = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3} # classes to numbers
        self.map_class = {v: k for k, v in self.class_map.items()} # numbers to classes

    def load_csv(self):
        """
        Load the CSV file into a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            # return pd.read_csv(self.csv_file_path)
            return df
        except FileNotFoundError:
            print(f"CSV file not found at {self.csv_file_path}. Creating an empty DataFrame.")
            return pd.DataFrame()

    def save_csv(self):
        """
        Save the current DataFrame back to the CSV file.
        """
        self.data.to_csv(self.csv_file_path, index=False)

    def add_record(self, record):
        """
        Add a new record to the DataFrame and save it to the CSV file.
        """
        self.data = pd.concat([self.data, pd.DataFrame([record])], ignore_index=True)
        self.save_csv()

    def delete_record(self, record_id):
        """
        Delete a record by its ID and save the updated DataFrame to the CSV file.
        """
        self.data = self.data[self.data['id'] != record_id]
        self.save_csv()

    def load_model(self):
        """
        Load the XGBoost model from the specified file path.
        """
        # try:
        #     return xgb.Booster(model_file=self.model_file_path)
        # except Exception as e:
        #     print(f"Error loading model: {e}")
        #     return None
        # load classifier model
        self.classifier_model = XGBClassifier()
        self.classifier_model.load_model(self.classifier_model_file_path)
        # load regressor model
        self.regressor_model = XGBRegressor()
        self.regressor_model.load_model(self.regressor_model_file_path)

    def predict(self, input_data):
        """
        Predict using the XGBoost model.
        :param input_data: A pandas DataFrame containing the input features.
        :return: Predictions as a numpy array.
        """
        # if self.model is None:
        #     raise ValueError("Model is not loaded.")
        
        # dmatrix = xgb.DMatrix(input_data)
        # return self.model.predict(dmatrix)
        #
        # Error handling
        if (self.classifier_model is None) or (self.regressor_model is None):
            raise ValueError("At least one model is not loaded.")
        # print(f"SECOND : {input_data}")

        # predict the integral of the tail and the max deviation between the tails of the real and ideal responses
        intR_maxdev = self.regressor_model.predict(input_data[self.input_columns])
        # print(f"THIRD : {predictions}")
        intR_maxdev_df = pd.DataFrame(intR_maxdev, columns=[['integral_R', 'max_deviation']])
        # print(f"FOURTH : {prediction_df}")
        # predict the class using the predicted integral and max deviation
        pred_class = self.classifier_model.predict(intR_maxdev_df)
        # print(f"FITH : {pred_class}")
        predicted_class = [self.map_class[pred] for pred in pred_class]
        return intR_maxdev_df, predicted_class
    
class ROOTmodel:
    def __init__(self, root_file_path: str, hist_prefix: str):
        self.root_file_path     = root_file_path
        self.hist_prefix        = hist_prefix
        self.rootdata, self.channels = self.read_ROOT(filename=self.root_file_path)


    def read_ROOT(self, filename: str):
        root_file       = uproot.open(filename)
        all_hist1d      = [f for f in root_file.keys() if self.hist_prefix in f]
        all_channels    = [int(hist1d.split(';')[0].split('_')[-1]) for hist1d in all_hist1d]
        return root_file, all_channels 

    def getCHN_resp(self, chn=0):
        histname        = '_'.join([self.hist_prefix, 'channel', f'{chn};1'])
        Npoints         = 70
        hist            = np.array([])
        try:
            hist        = self.rootdata[histname].to_numpy()[0][:Npoints]
            time_ticks  = self.rootdata[histname].to_numpy()[1][:Npoints]
        except:
            histname    = '_'.join([self.hist_prefix, 'channel', f'{chn};0'])
            hist        = self.rootdata[histname].to_numpy()[0][:Npoints]
            time_ticks  = self.rootdata[histname].to_numpy()[1][:Npoints]
        # wf              = hist.copy(
        # return wf#, hist
        wf      = [float(d) for d in hist]
        tticks  = [float(d) for d in time_ticks]
        return tticks, wf

