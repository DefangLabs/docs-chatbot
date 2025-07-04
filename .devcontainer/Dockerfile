FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

RUN pip install --no-cache-dir poetry
COPY ./app/pyproject.toml /app/pyproject.toml
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY ./app /app

