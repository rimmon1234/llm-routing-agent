#!/bin/bash
set -e

# Start Ollama in the background
echo "🦙 Starting local Ollama server in background..."
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama to be active and listening
echo "⏳ Waiting for Ollama server to boot..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/ >/dev/null; then
        echo "✅ Ollama is ready and listening."
        break
    fi
    sleep 1
done

# Run the evaluation runner
echo "🚀 Running the evaluation runner..."
exec python -u evaluation_runner.py "$@"
