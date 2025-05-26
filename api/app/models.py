import pandas as pd
import xgboost as xgb

class CSVModel:
    def __init__(self, csv_file_path, model_file_path):
        """
        Initialize the CSVModel with the path to the CSV file and the XGBoost model.
        """
        self.csv_file_path = csv_file_path
        self.model_file_path = model_file_path
        self.data = self.load_csv()
        # print(self.data[self.data['#Ch.#']==5])
        # self.model = self.load_model()

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
        try:
            return xgb.Booster(model_file=self.model_file_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, input_data):
        """
        Predict using the XGBoost model.
        :param input_data: A pandas DataFrame containing the input features.
        :return: Predictions as a numpy array.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        dmatrix = xgb.DMatrix(input_data)
        return self.model.predict(dmatrix)

# from flask_sqlalchemy import SQLAlchemy

# db = SQLAlchemy()

# class ExampleModel(db.Model):
#     __tablename__ = 'example_model'

#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     description = db.Column(db.String(255), nullable=True)

#     def __repr__(self):
#         return f'<ExampleModel {self.name}>'