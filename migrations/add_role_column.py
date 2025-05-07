# migrations/add_role_column.py
from alembic import op
import sqlalchemy as sa


def upgrade():
    op.add_column('user', sa.Column('role', sa.String(20), server_default='user'))
    op.add_column('user', sa.Column('last_login', sa.DateTime))
    op.add_column('user', sa.Column('login_count', sa.Integer, server_default='0'))

    # Migrate existing admins to have the admin role
    op.execute("UPDATE user SET role = 'admin' WHERE is_admin = TRUE")


def downgrade():
    op.drop_column('user', 'role')
    op.drop_column('user', 'last_login')
    op.drop_column('user', 'login_count')