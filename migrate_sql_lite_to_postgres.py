# migrate_sqlite_to_postgres.py
import sqlite3
import psycopg2
import os
import glob

# PostgreSQL connection parameters
PG_HOST = "phillyscript-db-1.ct2ce24yc54l.eu-west-2.rds.amazonaws.com"
PG_PORT = "5432"
PG_NAME = "postgres"
PG_USER = "phillyscripter"
PG_PASSWORD = "S)(BEz-rRL[2]7h5tc6;H%"


def find_sqlite_db():
    """Find the SQLite database file"""
    possible_locations = [
        './fallback.db',
        '../fallback.db',
        '/var/app/current/fallback.db',
        '/tmp/fallback.db'
    ]

    # Also search for any .db files
    db_files = glob.glob('**/*.db', recursive=True)
    possible_locations.extend(db_files)

    for location in possible_locations:
        if os.path.exists(location):
            print(f"Found database at: {location}")

            # Check if it has a user table
            try:
                conn = sqlite3.connect(location)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user';")
                if cursor.fetchone():
                    print(f"Database at {location} contains 'user' table")
                    conn.close()
                    return location
                conn.close()
            except Exception as e:
                print(f"Error examining database {location}: {e}")

    return None


def migrate_data(sqlite_db_path):
    """Migrate data from SQLite to PostgreSQL"""
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_NAME,
        user=PG_USER,
        password=PG_PASSWORD
    )
    pg_cursor = pg_conn.cursor()

    try:
        # Get column names from SQLite
        sqlite_cursor.execute("PRAGMA table_info(user);")
        columns_info = sqlite_cursor.fetchall()
        sqlite_columns = [col[1] for col in columns_info]

        # Get data from SQLite
        sqlite_cursor.execute(f"SELECT * FROM user;")
        users = sqlite_cursor.fetchall()

        if not users:
            print("No users found in SQLite database to migrate")
            return

        print(f"Found {len(users)} users to migrate")

        # Check which columns exist in PostgreSQL
        pg_cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'user'
        """)
        pg_columns = [col[0] for col in pg_cursor.fetchall()]

        # Find common columns that exist in both databases
        common_columns = [col for col in sqlite_columns if col.lower() in [pgcol.lower() for pgcol in pg_columns]]

        print(f"Common columns: {', '.join(common_columns)}")

        # Create placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(common_columns))

        # Insert data into PostgreSQL
        for user in users:
            # Map data from SQLite columns to PostgreSQL columns
            user_data = []
            for i, col in enumerate(sqlite_columns):
                if col in common_columns:
                    user_data.append(user[i])

            pg_cursor.execute(f"""
                INSERT INTO "user" ({', '.join([f'"{col}"' for col in common_columns])}) 
                VALUES ({placeholders})
                ON CONFLICT (username) DO NOTHING;
            """, user_data)

        pg_conn.commit()
        print(f"Successfully migrated {len(users)} users to PostgreSQL")

        # Add role column values based on is_admin
        pg_cursor.execute("""
            UPDATE "user" 
            SET role = 'admin' 
            WHERE is_admin = TRUE AND role = 'user';
        """)

        # Set the first user as master_admin
        pg_cursor.execute("""
            UPDATE "user" 
            SET role = 'master_admin' 
            WHERE id = (SELECT MIN(id) FROM "user" WHERE is_admin = TRUE)
            AND role = 'admin';
        """)

        pg_conn.commit()
        print("Updated user roles successfully")

    except Exception as e:
        pg_conn.rollback()
        print(f"Error during migration: {e}")
    finally:
        sqlite_cursor.close()
        sqlite_conn.close()
        pg_cursor.close()
        pg_conn.close()


if __name__ == "__main__":
    sqlite_db_path = find_sqlite_db()
    if sqlite_db_path:
        migrate_data(sqlite_db_path)
    else:
        print("No SQLite database with 'user' table found")