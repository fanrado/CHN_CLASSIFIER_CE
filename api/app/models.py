from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ExampleModel(db.Model):
    __tablename__ = 'example_model'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<ExampleModel {self.name}>'