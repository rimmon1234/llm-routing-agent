#!/bin/bash
set -e

# Ensure Ollama knows where the baked-in models are located
export OLLAMA_MODELS=/app/ollama_models

# Start Ollama in the background
echo "🦙 Starting local Ollama server in background..."
ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to be active and listening
echo "⏳ Waiting for Ollama server to boot..."
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-120}
START_TIME=$(date +%s)
while true; do
    if curl -s http://127.0.0.1:11434/ >/dev/null; then
        echo "✅ Ollama is ready and listening."
        break
    fi
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -ge $STARTUP_TIMEOUT ]; then
        echo "❌ Timeout waiting for Ollama server to start (exceeded ${STARTUP_TIMEOUT}s)."
        echo "=== Ollama Server Logs (/tmp/ollama.log) ==="
        cat /tmp/ollama.log || echo "Could not read /tmp/ollama.log"
        exit 1
    fi
    sleep 1
done

# Verify that the model is loaded and present
echo "📦 Available local models:"
ollama list

# Warm up Ollama model
LOCAL_MODEL=${LOCAL_MODEL:-llama3.2:3b}
echo "🔥 Warming up Ollama model '${LOCAL_MODEL}'..."
curl -s -m 90 -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${LOCAL_MODEL}\", \"prompt\": \"Hello\", \"stream\": false, \"keep_alive\": \"30m\", \"options\": {\"num_predict\": 1}}" > /dev/null || echo "⚠️ Warmup timed out or failed, continuing..."

# Run the evaluation runner
echo "🚀 Running the evaluation runner..."
exec python -u evaluation_runner.py "$@"
