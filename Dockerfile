# Use an official Python runtime as a parent image
FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/rimmon1234/llm-routing-agent"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (linux-amd64 binary)
RUN curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst \
    && tar --zstd -xf ollama.tar.zst -C /usr \
    && rm ollama.tar.zst

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download tiktoken vocabulary to prevent runtime download timeout in offline environment
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
RUN mkdir -p /app/tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
RUN chmod -R 777 /app/tiktoken_cache

# Copy the rest of the application code
COPY . .

# Pre-download the local model during the build stage so it is baked in
# This ensures container starts and is ready in under 5 seconds (Hackathon 60s rule)
ENV OLLAMA_MODELS=/app/ollama_models
RUN (ollama serve &) && sleep 5 && ollama pull llama3.2:3b
RUN chmod -R 777 /app/ollama_models

# Set default environment variables (overridden by harness at runtime)
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV LOCAL_MODEL=llama3.2:3b
ENV REMOTE_MODEL=accounts/fireworks/models/deepseek-v3p2
ENV ROUTER_DEBUG_TIMING=1
ENV HOME=/tmp
ENV DOCKER_CONTAINER=1

# Ensure entrypoint is executable and has Unix line endings (safe for Windows checkouts)
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Use entrypoint.sh to start Ollama in background and run evaluation_runner.py
ENTRYPOINT ["./entrypoint.sh"]
