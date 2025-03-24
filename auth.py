from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import User
from extensions import db

auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(username=username).first()

    # Check if the user exists and the password is correct
    if not user or not user.check_password(password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login'))

    # If the user is not active
    if not user.is_active:
        flash('Your account has been deactivated. Please contact an administrator.')
        return redirect(url_for('auth.login'))

    # If validation passes, log in the user
    login_user(user, remember=remember)

    # Get the page the user wanted to access before being redirected to login
    next_page = request.args.get('next')

    if not next_page or not next_page.startswith('/'):
        next_page = url_for('index')

    return redirect(next_page)


@auth.route('/signup')
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('signup.html')


@auth.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    # Check if passwords match
    if password != confirm_password:
        flash('Passwords do not match')
        return redirect(url_for('auth.signup'))

    # Check if username or email already exists
    user_by_email = User.query.filter_by(email=email).first()
    user_by_username = User.query.filter_by(username=username).first()

    if user_by_email:
        flash('Email address already exists')
        return redirect(url_for('auth.signup'))

    if user_by_username:
        flash('Username already exists')
        return redirect(url_for('auth.signup'))

    # Create a new user with the form data
    new_user = User(email=email, username=username)
    new_user.set_password(password)

    # Only first user is admin
    if User.query.count() == 0:
        new_user.is_admin = True

    # Add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    flash('Account created successfully! You can now log in.')
    return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@auth.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@auth.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # If a new password is provided
    if new_password:
        # Verify current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect')
            return redirect(url_for('auth.profile'))

        # Check if new passwords match
        if new_password != confirm_password:
            flash('New passwords do not match')
            return redirect(url_for('auth.profile'))

        # Update password
        current_user.set_password(new_password)
        db.session.commit()
        flash('Password updated successfully')

    return redirect(url_for('auth.profile'))


# Admin-only route example
@auth.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have permission to access the admin area')
        return redirect(url_for('index'))

    users = User.query.all()
    return render_template('admin.html', users=users)


@auth.route('/admin/toggle_user/<int:user_id>')
@login_required
def toggle_user(user_id):
    if not current_user.is_admin:
        flash('You do not have permission to perform this action')
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)

    # Don't allow deactivating your own account
    if user.id == current_user.id:
        flash('You cannot deactivate your own account')
        return redirect(url_for('auth.admin'))

    user.is_active = not user.is_active
    db.session.commit()

    status = "activated" if user.is_active else "deactivated"
    flash(f'User {user.username} has been {status}')

    return redirect(url_for('auth.admin'))