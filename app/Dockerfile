# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including Git, GCC, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Go for x86 architecture (latest supported version 1.21)
RUN curl -OL https://golang.org/dl/go1.21.1.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.1.linux-amd64.tar.gz && \
    rm go1.21.1.linux-amd64.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Install uWSGI and GitPython
RUN pip install uwsgi gitpython

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for the sentence transformers model
ENV SENTENCE_TRANSFORMERS_HOME="/root/.cache/sentence_transformers"

# Preload the sentence transformer model to cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application source code into the container
COPY . /app

# Expose port 5050 for the Flask application
EXPOSE 5050

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application using uWSGI
CMD ["uwsgi", "--lazy-apps", "--http", "0.0.0.0:5050", "--wsgi-file", "app.py", "--callable", "app", "--processes", "4", "--threads", "2"]
