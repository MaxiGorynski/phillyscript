import psycopg2
from datetime import datetime

# Database connection parameters
DB_HOST = "phillyscript-db-1.ct2ce24yc54l.eu-west-2.rds.amazonaws.com"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "phillyscripter"
DB_PASSWORD = "S)(BEz-rRL[2]7h5tc6;H%"


def migrate_usage_tracking():
    print("Starting migration to add usage tracking columns...")
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

        # Add new columns to user table
        cursor.execute("""
        ALTER TABLE "user" 
        ADD COLUMN IF NOT EXISTS total_transcription_minutes FLOAT DEFAULT 0.0,
        ADD COLUMN IF NOT EXISTS current_month_transcription_minutes FLOAT DEFAULT 0.0,
        ADD COLUMN IF NOT EXISTS last_usage_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)

        # Check if columns were added successfully
        cursor.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'user' AND column_name IN 
        ('total_transcription_minutes', 'current_month_transcription_minutes', 'last_usage_reset');
        """)

        added_columns = cursor.fetchall()
        print(f"Successfully added columns: {[col[0] for col in added_columns]}")

        conn.commit()
        print("Migration completed successfully.")

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
    migrate_usage_tracking()