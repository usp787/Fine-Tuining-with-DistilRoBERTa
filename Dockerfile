# Production Dockerfile for LoRA Emotion Classifier API
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY best_LoRA_model.pt .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 (AWS Lambda expects this, but you can change for EC2)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables (can be overridden at runtime)
ENV MODEL_PATH=best_LoRA_model.pt
ENV LORA_RANK=16
ENV DEVICE=cuda

# Run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
