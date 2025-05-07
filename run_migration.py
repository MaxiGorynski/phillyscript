# run_migration.py
import sqlite3
import os

# Update this path to point to your SQLite database file
DB_PATH = "instance/users.db"  # Adjust this to your actual database path


def run_migration():
    print(f"Connecting to database at: {DB_PATH}")

    # Check if the database file exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if role column exists
        cursor.execute("PRAGMA table_info(user)")
        columns = [info[1] for info in cursor.fetchall()]

        # Add the role column if it doesn't exist
        if 'role' not in columns:
            print("Adding 'role' column...")
            cursor.execute("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'")

            # Update existing admins to have the admin role
            print("Updating admin roles...")
            cursor.execute("UPDATE user SET role = 'admin' WHERE is_admin = 1")

            # Make first user the master_admin
            print("Setting first user as master_admin...")
            cursor.execute("UPDATE user SET role = 'master_admin' WHERE id = (SELECT MIN(id) FROM user)")
        else:
            print("Column 'role' already exists.")

        # Add last_login column if it doesn't exist
        if 'last_login' not in columns:
            print("Adding 'last_login' column...")
            cursor.execute("ALTER TABLE user ADD COLUMN last_login TIMESTAMP")
        else:
            print("Column 'last_login' already exists.")

        # Add login_count column if it doesn't exist
        if 'login_count' not in columns:
            print("Adding 'login_count' column...")
            cursor.execute("ALTER TABLE user ADD COLUMN login_count INTEGER DEFAULT 0")
        else:
            print("Column 'login_count' already exists.")

        # Commit the changes
        conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()