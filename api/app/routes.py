from flask import Blueprint, jsonify, request
from app.models import CSVModel
import pandas as pd
from flask_cors import CORS

api = Blueprint('api', __name__)
CORS(api, resources={r"/data/*": {"origins": "http://localhost:3000"}})
# CORS(api)
# Initialize the CSVModel
csv_model = CSVModel(csv_file_path='data/fit_results_run_30360_no_avg_labelled_tails.csv', model_file_path='xgboost_model.json')

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@api.route('/data/<int:data_id>', methods=['GET'])
def get_data(data_id):
    """
        Retrieve a record by its ID from the csv file.
    """
    try:
        record = csv_model.data[csv_model.data['#Ch.#']==int(data_id)]
        print(record)
        if record.empty:
            return jsonify({"error": "Record not found"}), 404
        return jsonify(record.to_dict(orient='records')[0]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@api.route('/predict', methods=['POST'])
def predict():
    """
    Predict using the XGBoost model with input data from the frontend.
    """
    input_data = request.json
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        input_df = pd.DataFrame([input_data])  # Convert input data to a DataFrame
        predictions = csv_model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500