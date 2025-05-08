from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, UserRole, db
from datetime import datetime
from auth_utils import admin_required, master_admin_required

auth = Blueprint('auth', __name__)


def backup_database():
    """Backup the database if possible"""
    if hasattr(current_app, 'backup_db_to_s3'):
        current_app.backup_db_to_s3()


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

    # Update login statistics
    user.login_count += 1
    user.last_login = datetime.utcnow()
    db.session.commit()

    # If validation passes, log in the user
    login_user(user, remember=remember)

    # Get the page the user wanted to access before being redirected to login
    next_page = request.args.get('next')

    if not next_page or not next_page.startswith('/'):
        # Redirect admins to appropriate page
        if user.is_master_admin():
            next_page = url_for('auth.master_admin_dashboard')
        elif user.is_admin_user:
            next_page = url_for('auth.admin')
        else:
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

    # Only first user gets the master_admin role
    if User.query.count() == 0:
        new_user.role = UserRole.MASTER_ADMIN.value
        new_user.is_admin = True

    # Add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    # Backup the database
    backup_database()

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

    db.session.commit()
    backup_database()

    return redirect(url_for('auth.profile'))


# Admin-only route example (using our backwards compatibility property)
@auth.route('/admin')
@login_required
def admin():
    if not current_user.is_admin_user:
        flash('You do not have permission to access the admin area')
        return redirect(url_for('index'))

    users = User.query.all()
    return render_template('admin.html', users=users)


@auth.route('/admin/toggle_user/<int:user_id>')
@login_required
def toggle_user(user_id):
    if not current_user.is_admin_user:
        flash('You do not have permission to perform this action')
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)

    # Master admins can toggle anyone
    # Regular admins can't toggle master admins or other admins
    if not current_user.is_master_admin() and (user.is_master_admin() or user.is_admin_user):
        flash('You do not have permission to modify this user')
        return redirect(url_for('auth.admin'))

    # Don't allow deactivating your own account
    if user.id == current_user.id:
        flash('You cannot deactivate your own account')
        return redirect(url_for('auth.admin'))

    user.is_active = not user.is_active
    db.session.commit()
    backup_database()

    status = "activated" if user.is_active else "deactivated"
    flash(f'User {user.username} has been {status}')

    return redirect(url_for('auth.admin'))


# New Master Admin Dashboard
@auth.route('/master-admin')
@master_admin_required
def master_admin_dashboard():
    users = User.query.all()

    # Calculate some basic stats
    user_count = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    admin_count = User.query.filter(User.role.in_([UserRole.ADMIN.value, UserRole.MASTER_ADMIN.value])).count()

    stats = {
        'total_users': user_count,
        'active_users': active_users,
        'admin_count': admin_count
    }

    return render_template('master_admin_dashboard.html', users=users, stats=stats)


@auth.route('/master-admin/promote/<int:user_id>')
@master_admin_required
def promote_user(user_id):
    user = User.query.get_or_404(user_id)

    # Don't allow promoting master admins
    if user.is_master_admin():
        flash('User is already a Master Admin')
        return redirect(url_for('auth.master_admin_dashboard'))

    # Promote regular user to admin
    if user.role == UserRole.USER.value:
        user.role = UserRole.ADMIN.value
        user.is_admin = True
        flash(f'User {user.username} has been promoted to Admin')

    # Promote admin to master admin
    elif user.role == UserRole.ADMIN.value:
        user.role = UserRole.MASTER_ADMIN.value
        flash(f'User {user.username} has been promoted to Master Admin')

    db.session.commit()
    backup_database()

    return redirect(url_for('auth.master_admin_dashboard'))


@auth.route('/master-admin/demote/<int:user_id>')
@master_admin_required
def demote_user(user_id):
    user = User.query.get_or_404(user_id)

    # Don't allow demoting yourself
    if user.id == current_user.id:
        flash('You cannot demote yourself')
        return redirect(url_for('auth.master_admin_dashboard'))

    # Demote master admin to admin
    if user.role == UserRole.MASTER_ADMIN.value:
        user.role = UserRole.ADMIN.value
        flash(f'User {user.username} has been demoted to Admin')

    # Demote admin to regular user
    elif user.role == UserRole.ADMIN.value:
        user.role = UserRole.USER.value
        user.is_admin = False
        flash(f'User {user.username} has been demoted to regular User')

    db.session.commit()
    backup_database()

    return redirect(url_for('auth.master_admin_dashboard'))


@auth.route('/create_user', methods=['POST'])
@login_required
def create_user():
    # Check if the current user has permission (is master admin)
    if not current_user.is_master_admin():
        flash('You do not have permission to create users.', 'danger')
        return redirect(url_for('master_admin_dashboard'))

    # Get form data
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')

    # Validate input
    if not all([username, email, password, role]):
        flash('All fields are required.', 'danger')
        return redirect(url_for('master_admin_dashboard'))

    # Check if username or email already exists
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('master_admin_dashboard'))

    if User.query.filter_by(email=email).first():
        flash('Email already exists.', 'danger')
        return redirect(url_for('master_admin_dashboard'))

    # Create new user
    new_user = User(
        username=username,
        email=email,
        role=role,
        is_active=True,
        is_admin=role in ['admin', 'master_admin'],
        created_at=datetime.utcnow()
    )
    new_user.set_password(password)

    # Add to database
    db.session.add(new_user)
    db.session.commit()

    flash(f'User {username} created successfully.', 'success')
    return redirect(url_for('master_admin_dashboard'))


@auth.route('/delete_user/<int:user_id>', methods=['GET'])
@login_required
def delete_user(user_id):
    # Check if the current user has permission (is master admin)
    if not current_user.is_master_admin():
        flash('You do not have permission to delete users.', 'danger')
        return redirect(url_for('auth.master_admin_dashboard'))

    # Find the user to delete
    user_to_delete = User.query.get_or_404(user_id)

    # Prevent deleting yourself
    if user_to_delete.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('auth.master_admin_dashboard'))

    # Get username before deletion for confirmation message
    username = user_to_delete.username

    # Delete the user
    db.session.delete(user_to_delete)
    db.session.commit()

    flash(f'User {username} has been deleted.', 'success')
    return redirect(url_for('master_admin_dashboard'))