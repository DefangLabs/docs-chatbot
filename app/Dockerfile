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

# Install Go for ARM architecture (latest supported version 1.21)
RUN curl -OL https://golang.org/dl/go1.21.1.linux-arm64.tar.gz && \
    tar -C /usr/local -xzf go1.21.1.linux-arm64.tar.gz && \
    rm go1.21.1.linux-arm64.tar.gz

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

# Copy the application source code into the container
COPY . /app

# Make the prebuild.sh script executable
RUN chmod +x .tmp/prebuild.sh

# Expose port 5050 for the Flask application
EXPOSE 5050

# Run test file
RUN python test_intercom.py

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application using uWSGI
CMD ["uwsgi", "--http", "0.0.0.0:5050", "--wsgi-file", "app.py", "--callable", "app", "--processes", "4", "--threads", "2"]
