# syntax=docker/dockerfile:1.4
# Cache mount for pip (build) avoids temp-dir cleanup failures under Buildx / QEMU
# (OSError: [Errno 30] Read-only file system on .whl during pip cleanup)

FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TMPDIR=/tmp

# Install Python dependencies
COPY ml/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade "pip>=24.2" "setuptools>=70" "wheel" && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy ML source code
COPY ml/ .

# Create data and models directories
RUN mkdir -p data models

EXPOSE 5000

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["python", "flask_api.py"]
