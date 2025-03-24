from flask import Flask
import os
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path

# Initialize Flask and SQLAlchemy
application = Flask(__name__)
application.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-local-only')
application.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(application)

# Define a simple User model for testing
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))

# Create directories for testing file operations
UPLOAD_FOLDER = Path('/app/temp_uploads')
TRANSCRIPT_FOLDER = Path('/app/temp_transcripts')
RESULT_FOLDER = Path('/app/static/results')

for folder in [UPLOAD_FOLDER, TRANSCRIPT_FOLDER, RESULT_FOLDER]:
    try:
        folder.mkdir(exist_ok=True, parents=True)
        print(f"Created directory: {folder}")
    except Exception as e:
        print(f"Error creating {folder}: {e}")

# Create database tables
with application.app_context():
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Database error: {e}")

@application.route('/')
def home():
    return "Test app running with database and file operations!"

@application.route('/test-db')
def test_db():
    try:
        # Test DB query
        user_count = User.query.count()
        return f"Database connection successful. User count: {user_count}"
    except Exception as e:
        return f"Database error: {str(e)}"

@application.route('/test-files')
def test_files():
    try:
        # Test file operations
        test_file = UPLOAD_FOLDER / "test.txt"
        with open(test_file, "w") as f:
            f.write("Test file")
        return f"File operations successful. Created {test_file}"
    except Exception as e:
        return f"File error: {str(e)}"

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8080)