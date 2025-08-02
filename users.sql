-- User table data from AWS RDS export
-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    created_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    role VARCHAR(50),
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    total_transcription_minutes FLOAT DEFAULT 0.0,
    current_month_transcription_minutes FLOAT DEFAULT 0.0,
    last_usage_reset TIMESTAMP
);

-- Insert user data
INSERT INTO users (id, username, email, password_hash, created_at, is_active, is_admin, role, last_login, login_count, total_transcription_minutes, current_month_transcription_minutes, last_usage_reset) VALUES (4, 'Maxi', 'm.gorynski@yahoo.co.uk', 'pbkdf2:sha256:260000$IfwlAMWWjYscezSx$deedb69fcdf560a21507a0000baa1464bb27d9770155bef0f6c6e1b58a8d555f', '2025-05-09 13:40:46.405234', TRUE, FALSE, 'user', '2025-05-09 13:49:30.171548', 1, 8.8635, 8.8635, '2025-05-09 13:40:46.588166');
INSERT INTO users (id, username, email, password_hash, created_at, is_active, is_admin, role, last_login, login_count, total_transcription_minutes, current_month_transcription_minutes, last_usage_reset) VALUES (1, 'admin', 'admin@example.com', 'pbkdf2:sha256:260000$R9Sd0rqwnID23XZV$03416f9c75ce2dd27109e96df6b07e043606c832e456d497547d63182428b286', '2025-05-07 15:59:44.75595', TRUE, TRUE, 'master_admin', '2025-05-28 17:32:06.324932', 13, 1.47725, 1.47725, '2025-05-08 16:16:22.225421');
