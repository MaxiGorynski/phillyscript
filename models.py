from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import os
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
import uuid
import cv2
import numpy as np
import uuid
from werkzeug.utils import secure_filename
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import difflib
import os
from werkzeug.utils import secure_filename
import uuid
import re
from docx import Document
from io import BytesIO
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'