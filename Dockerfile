FROM python:3.10-slim

WORKDIR /app

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=application.py

EXPOSE 8080

CMD ["gunicorn", "--config", "gunicorn.conf.py", "application:application"]
