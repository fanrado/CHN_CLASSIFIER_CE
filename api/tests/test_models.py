from app import create_app, db
from app.models import YourModel  # Replace with your actual model
import pytest

@pytest.fixture
def app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def test_your_model_creation(client):
    # Example test for model creation
    model_instance = YourModel(field1='value1', field2='value2')  # Replace with actual fields
    db.session.add(model_instance)
    db.session.commit()
    
    assert model_instance.id is not None  # Check if the instance has an ID after commit

def test_your_model_query(client):
    # Example test for querying the model
    model_instance = YourModel(field1='value1', field2='value2')  # Replace with actual fields
    db.session.add(model_instance)
    db.session.commit()
    
    queried_instance = YourModel.query.filter_by(field1='value1').first()
    assert queried_instance is not None
    assert queried_instance.field2 == 'value2'  # Check if the queried instance has the correct field value