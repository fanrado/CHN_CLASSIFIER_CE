from flask import json
from app import create_app

app = create_app()

def test_home_route():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert b'Welcome to the Flask API' in response.data

def test_some_route():
    response = app.test_client().get('/some-route')
    assert response.status_code == 200
    assert json.loads(response.data) == {'message': 'This is some route'}