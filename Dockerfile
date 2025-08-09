FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY railway-requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r railway-requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/transcriptions /app/data/uploads /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD cd api && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
