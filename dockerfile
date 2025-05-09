# Use Python 3.8 as the base image
FROM python:3.8

# Set working directory
WORKDIR /app

# Install system dependencies (including audio processing libraries and PostgreSQL)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    flac \
    swig \
    postgresql-client \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=application.py

# Expose port for the application
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "application:application"]