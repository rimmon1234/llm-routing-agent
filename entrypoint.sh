#!/bin/bash
set -e

# Start Ollama in the background
echo "🦙 Starting local Ollama server in background..."
ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to be active and listening
echo "⏳ Waiting for Ollama server to boot..."
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-55}
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
        exit 1
    fi
    sleep 1
done

# Run the evaluation runner
echo "🚀 Running the evaluation runner..."
exec python -u evaluation_runner.py "$@"
