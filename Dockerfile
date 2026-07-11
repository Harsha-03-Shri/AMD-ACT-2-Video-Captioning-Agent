# Build with: docker build --platform linux/amd64 -t video-captioning-agent .
FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies (ffmpeg for keyframe extraction fallback)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Server-mode API wrapper (Render) — lives at workspace root alongside app/
COPY app_server.py .

# Copy model configuration
COPY models_config.json .

# API keys — replace with your actual keys before building



# Default mount points for batch mode
RUN mkdir -p /input /output
 
ENV PYTHONUNBUFFERED=1
 
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh
 
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
 
