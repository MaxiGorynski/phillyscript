import psycopg2
from werkzeug.security import generate_password_hash
from datetime import datetime

# Database connection parameters
DB_HOST = "phillyscript-db-1.ct2ce24yc54l.eu-west-2.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "phillyscripter"
DB_PASSWORD = "S)(BEz-rRL[2]7h5tc6;H%"


def setup_postgres_db():
    # Connect to the database
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        conn.autocommit = False

        # Create the user table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS "user" (
            id SERIAL PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            is_admin BOOLEAN DEFAULT FALSE,
            role VARCHAR(20) DEFAULT 'user',
            last_login TIMESTAMP,
            login_count INTEGER DEFAULT 0
        );
        """)

        # Check if we need to create a master admin user
        cursor.execute("SELECT COUNT(*) FROM \"user\";")
        user_count = cursor.fetchone()[0]

        if user_count == 0:
            # Create a master admin user if no users exist
            now = datetime.utcnow()
            admin_password_hash = generate_password_hash("changeme123")  # Change this!

            cursor.execute("""
            INSERT INTO "user" (username, email, password_hash, created_at, is_active, is_admin, role)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, ('admin', 'admin@example.com', admin_password_hash, now, True, True, 'master_admin'))

            print("Created master admin user: admin@example.com")

        # Commit the changes
        conn.commit()
        print("Database setup completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    setup_postgres_db()