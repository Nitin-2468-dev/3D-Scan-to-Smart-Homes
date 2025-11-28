FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY api/requirements.txt /app/api/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy application code
COPY api/ /app/api/
COPY fallback/ /app/fallback/

# Create directories for uploads and artifacts
RUN mkdir -p /app/uploads /app/artifacts

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
