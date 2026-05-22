# ==================== Multi-stage Dockerfile ====================
# Production-ready Docker image for Product Recommender System

# ==================== Stage 1: Builder ====================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir pip==23.3.1 setuptools==65.5.0 wheel==0.41.2 && \
    pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime ====================
FROM python:3.10-slim

# Set labels
LABEL maintainer="your.email@example.com"
LABEL version="1.0.0"
LABEL description="Product Recommender System API"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/synthetic \
    /app/models \
    /app/logs \
    /app/mlruns

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY frontend/ /app/frontend/
COPY data/synthetic/ /app/data/synthetic/

# Copy configuration files
COPY setup.py /app/
COPY README.md /app/
COPY requirements.txt /app/

# Set python path to find src module
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]