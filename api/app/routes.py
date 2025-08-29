from flask import Blueprint, jsonify, request
from app.models import CSVModel, ROOTmodel
import pandas as pd
import numpy as np
from flask_cors import CORS

api = Blueprint('api', __name__)
CORS(api, resources={r"/data/*": {"origins": "http://localhost:3000"}})
# CORS(api)
# Initialize the CSVModel
# csv_model = CSVModel(csv_file_path='data/fit_results_run_30360_no_avg_labelled_tails.csv', model_file_path='xgboost_model.json')
# csv_model   = CSVModel(csv_file_path='data/fit_results_run_30360_no_avg_labelled_tails.csv', regressor_model_file_path='models/regressor_bdt_model.json', classifier_model_file_path='models/classifier_bdt_model.json')
csv_model   = CSVModel(csv_file_path='data/fit_results_run_30413_no_avg_labelled_tails.csv', regressor_model_file_path='models/regressor_bdt_model.json', classifier_model_file_path='models/classifier_bdt_model.json')
root_model  = ROOTmodel(root_file_path='data/raw_waveforms_run_30413.root', hist_prefix='hist_0') # read the root data

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@api.route('/data/<int:data_id>', methods=['GET'])
def get_data(data_id):
    """
        Retrieve a record by its ID from the csv file.
    """
    class_map = {'c1': 'undershoot only',
                 'c2': 'undershoot with samll overshoot',
                 'c3': 'overshoot with small undershoot',
                 'c4': 'overshoot only'}
    try:
        chn     = int(data_id)
        # record  = csv_model.data[csv_model.data['#Ch.#']==int(data_id)]
        record  = csv_model.data[csv_model.data['#Ch.#']==chn]
        # print(record) #
        if record.empty:
            return jsonify({"error": "Record not found"}), 404
        intR_maxdev, pred_class         = csv_model.predict(record)
        idx                             = record.index[0]
        record.loc[idx, 'integral_R']   = np.round(intR_maxdev.iloc[0]['integral_R'], 4)
        record.loc[idx, 'max_deviation'] = np.round(intR_maxdev.iloc[0]['max_deviation'], 4)
        record.loc[idx, 'class']        = class_map[pred_class[0]]
        record_dict                     = record.to_dict(orient='records')[0]
        tticks, wf                      = root_model.getCHN_resp(chn=chn)
        record_dict['realResp']         = wf
        record_dict['timeticks']        = tticks
        # return jsonify(record.to_dict(orient='records')[0]), 200
        return jsonify(record_dict), 200
        # return json_record, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500