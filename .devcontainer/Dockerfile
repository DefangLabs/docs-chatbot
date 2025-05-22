FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

COPY ./app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY ./app /app
