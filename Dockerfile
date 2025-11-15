# Multi-stage build for SpiralMind-Nexus
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r spiral && useradd -r -g spiral spiral

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R spiral:spiral /app

# Switch to non-root user
USER spiral

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD spiral --text "Health check" --format minimal || exit 1

# Default command
CMD ["spiral", "--help"]

# Production stage
FROM base as production

# Set production environment
ENV SPIRAL_ENV=production

# Expose port for API
EXPOSE 8000

# API command
CMD ["python", "-m", "spiral.api", "--host", "0.0.0.0", "--port", "8000"]

# CLI stage
FROM base as cli

# Set CLI environment
ENV SPIRAL_ENV=production

# CLI command
ENTRYPOINT ["spiral"]
CMD ["--help"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Set development environment
ENV SPIRAL_ENV=development

# Switch back to root for development tools
USER root

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Switch back to spiral user
USER spiral

# Development command
CMD ["bash"]
