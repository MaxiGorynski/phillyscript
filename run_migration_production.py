# run_migration_postgres.py
import psycopg2
import os

# Update these with your PostgreSQL connection details
DB_HOST = "phillyscript-db-1.ct2ce24yc54l.eu-west-2.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "postgres"  # Update this if you're using a different database name
DB_USER = "your_db_username"  # Update with your database username
DB_PASSWORD = "your_db_password"  # Update with your database password


def run_migration():
    print(f"Connecting to database at: {DB_HOST}")

    # Connect to the database
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        conn.autocommit = False  # Use transactions

        # Check if role column exists
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'user'")
        columns = [column[0] for column in cursor.fetchall()]

        # Add the role column if it doesn't exist
        if 'role' not in columns:
            print("Adding 'role' column...")
            cursor.execute("ALTER TABLE \"user\" ADD COLUMN role VARCHAR(20) DEFAULT 'user'")

            # Update existing admins to have the admin role
            print("Updating admin roles...")
            cursor.execute("UPDATE \"user\" SET role = 'admin' WHERE is_admin = TRUE")

            # Make first user the master_admin
            print("Setting first user as master_admin...")
            cursor.execute("UPDATE \"user\" SET role = 'master_admin' WHERE id = (SELECT MIN(id) FROM \"user\")")
        else:
            print("Column 'role' already exists.")

        # Add last_login column if it doesn't exist
        if 'last_login' not in columns:
            print("Adding 'last_login' column...")
            cursor.execute("ALTER TABLE \"user\" ADD COLUMN last_login TIMESTAMP")
        else:
            print("Column 'last_login' already exists.")

        # Add login_count column if it doesn't exist
        if 'login_count' not in columns:
            print("Adding 'login_count' column...")
            cursor.execute("ALTER TABLE \"user\" ADD COLUMN login_count INTEGER DEFAULT 0")
        else:
            print("Column 'login_count' already exists.")

        # Commit the changes
        conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    run_migration()