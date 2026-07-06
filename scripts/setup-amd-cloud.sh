#!/usr/bin/env bash
# ============================================================
# AMD Developer Cloud — One-Click Setup for Hybrid LLM Router
# ============================================================
# This script automates the full setup of the Hybrid LLM
# Routing Agent on an AMD Developer Cloud instance (or any
# Linux machine with AMD ROCm).
#
# Usage:
#   chmod +x scripts/setup-amd-cloud.sh
#   ./scripts/setup-amd-cloud.sh
#
# Or for a dry run (prints steps without executing):
#   ./scripts/setup-amd-cloud.sh --dry-run
# ============================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN — Commands will be printed but not executed."
    echo ""
fi

run() {
    if $DRY_RUN; then
        echo "  ▶ $*"
    else
        echo "  ▶ $*"
        "$@"
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   AMD Developer Cloud — Hybrid LLM Router Setup ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Detect AMD GPU ──────────────────────────────
echo "🔎 Step 0: Detecting AMD GPU..."
if command -v rocminfo &>/dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -1 | awk -F': ' '{print $2}')
    if [[ -n "$GPU_NAME" ]]; then
        echo "   ✅ AMD GPU detected: $GPU_NAME"
    else
        echo "   ⚠️  rocminfo found but no AMD GPU detected. Continuing in CPU mode."
    fi
else
    echo "   ⚠️  rocminfo not found. Install ROCm first or continue in CPU mode."
fi
echo ""

# ── Step 1: System dependencies ──────────────────────────
echo "📦 Step 1: Installing system dependencies..."
run sudo apt-get update -qq
run sudo apt-get install -y -qq python3 python3-pip python3-venv curl git
echo ""

# ── Step 2: Install Ollama ───────────────────────────────
echo "🦙 Step 2: Installing Ollama (ROCm-enabled)..."
if command -v ollama &>/dev/null; then
    echo "   ✅ Ollama already installed ($(ollama --version))"
else
    run curl -fsSL https://ollama.com/install.sh | sh
fi
echo ""

# ── Step 3: Configure Ollama for AMD MI300X ──────────────
echo "⚙️  Step 3: Configuring Ollama for AMD GPU..."
if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -q "gfx942"; then
    echo "   🎯 MI300X detected — setting HSA_OVERRIDE_GFX_VERSION"
    OLLAMA_SERVICE_DIR="/etc/systemd/system/ollama.service.d"
    run sudo mkdir -p "$OLLAMA_SERVICE_DIR"
    if ! $DRY_RUN; then
        sudo tee "$OLLAMA_SERVICE_DIR/override.conf" > /dev/null << 'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=9.4.2"
Environment="OLLAMA_HOST=0.0.0.0"
EOF
    fi
    run sudo systemctl daemon-reload
    run sudo systemctl restart ollama
    echo "   ✅ Ollama configured for MI300X"
else
    echo "   ℹ️  No MI300X detected — using default Ollama config"
    run sudo systemctl restart ollama 2>/dev/null || echo "   ℹ️  Starting ollama..."
fi
echo ""

# ── Step 4: Pull the local model ─────────────────────────
echo "📥 Step 4: Pulling local model (llama3.2:3b)..."
run ollama pull llama3.2:3b
echo ""

# ── Step 5: Clone repo & install Python deps ─────────────
echo "🐍 Step 5: Setting up Python environment..."
if [[ ! -d "llm-routing-agent" ]]; then
    # If we're already in the repo directory, skip cloning
    if [[ -f "main.py" ]] && [[ -f "requirements.txt" ]]; then
        echo "   ✅ Already in project directory"
    else
        echo "   ⚠️  Project not found. Clone manually:"
        echo "   git clone https://github.com/yourusername/llm-routing-agent.git"
        echo "   Then re-run this script from the project root."
        exit 1
    fi
fi
run python3 -m venv .venv
run source .venv/bin/activate
run pip install -q -r requirements.txt
echo ""

# ── Step 6: Configure environment ────────────────────────
echo "🔑 Step 6: Configuring environment..."
if [[ ! -f ".env" ]]; then
    run cp .env.example .env
    echo ""
    echo "   ⚠️  Edit .env to add your FIREWORKS_API_KEY:"
    echo "      nano .env"
    echo "   (Agent will work without it, but remote fallback will be mocked.)"
else
    echo "   ✅ .env already exists"
fi
echo ""

# ── Step 7: Run smoke test ───────────────────────────────
echo "🧪 Step 7: Running smoke test..."
run python3 main.py --clear-cache 2>/dev/null
run python3 -c "
from src.client import LLMClient
c = LLMClient()
gpu = c.gpu_info
if gpu['available']:
    print(f'   ✅ AMD GPU confirmed: {gpu[\"name\"]} ({gpu[\"vram_gb\"]} GB)')
else:
    print('   ℹ️  Running on CPU (AMD GPU not detected)')
print(f'   ✅ Ollama reachable: {c.call_local(\"ping\", max_tokens=2)}')
"
echo ""

# ── Done ─────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete!                             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. (Optional) Edit .env with your Fireworks API key"
echo "  2. Run the test suite:  python3 test_router.py"
echo "  3. Try interactive mode: python3 main.py --strategy fallback"
echo "  4. Run a single query:   python3 main.py --query \"Your query\""
echo ""
echo "Happy routing! 🧠✨"
