# auth_utils.py
from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('auth.login', next=request.url))

            if not current_user.has_role(role):
                flash('You do not have permission to access this page.')
                return redirect(url_for('index'))

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# Specific decorators
def admin_required(f):
    return role_required('admin')(f)


def master_admin_required(f):
    return role_required('master_admin')(f)