from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from extensions import db
from enum import Enum


class UserRole(Enum):
    USER = 'user'
    ADMIN = 'admin'
    MASTER_ADMIN = 'master_admin'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)  # Keeping for backwards compatibility
    role = db.Column(db.String(20), default=UserRole.USER.value)

    # Additional fields for tracking
    last_login = db.Column(db.DateTime)
    login_count = db.Column(db.Integer, default=0)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Role checking methods
    def has_role(self, role):
        if isinstance(role, UserRole):
            return self.role == role.value
        return self.role == role

    def is_master_admin(self):
        return self.role == UserRole.MASTER_ADMIN.value

    # For backward compatibility
    @property
    def is_admin_user(self):
        return self.is_admin or self.role in (UserRole.ADMIN.value, UserRole.MASTER_ADMIN.value)

    def __repr__(self):
        return f'<User {self.username}>'