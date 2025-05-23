from flask import Blueprint, jsonify, request

api = Blueprint('api', __name__)

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@api.route('/data', methods=['POST'])
def create_data():
    data = request.json
    # Here you would typically process the data and save it to the database
    return jsonify({"message": "Data created", "data": data}), 201

@api.route('/data/<int:data_id>', methods=['GET'])
def get_data(data_id):
    # Here you would typically retrieve the data from the database
    return jsonify({"data_id": data_id, "data": "Sample data"}), 200

@api.route('/data/<int:data_id>', methods=['DELETE'])
def delete_data(data_id):
    # Here you would typically delete the data from the database
    return jsonify({"message": "Data deleted", "data_id": data_id}), 204