# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:8080"

# Worker processes
workers = 2
worker_class = "sync"
threads = 1

# Timeouts - critical for preventing worker timeouts
timeout = 120  # Increase from default 30s to 120s

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Reduce server load
keepalive = 5