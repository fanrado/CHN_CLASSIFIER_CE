# API

This is the API for the channel response classification development. All back-end operations will be included in this app.

## Project Structure

```
api
├── app
│   ├── __init__.py          # Initializes the Flask application
│   ├── routes.py            # Defines API routes
│   ├── models.py            # Contains data models
│   ├── services              # Business logic or service layer
│   │   └── __init__.py
│   ├── utils                 # Helper functions and classes
│   │   └── __init__.py
│   └── config.py            # Configuration settings
├── tests                     # Unit tests for the application
│   ├── __init__.py
│   ├── test_routes.py       # Tests for route handlers
│   └── test_models.py       # Tests for data models
├── requirements.txt          # Project dependencies
├── wsgi.py                   # Entry point for WSGI servers
├── .env                      # Environment variables
├── .gitignore                # Files to ignore in Git
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your environment-specific variables.

5. **Run the application:**
   ```
   python wsgi.py
   ```

## Usage

- The API is accessible at `http://localhost:5000`.
- Use tools like Postman or curl to interact with the API endpoints defined in `app/routes.py`.

## Testing

- To run the tests, ensure your virtual environment is activated and run:
  ```
  pytest
  ```