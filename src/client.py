import os
import re
import subprocess
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        # Local client configuration (Ollama)
        local_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip('/')
        self.local_client = OpenAI(
            base_url=f"{local_host}/v1",
            api_key="ollama",  # Ollama doesn't require a key but OpenAI client expects a non-empty string
            timeout=20.0       # Timeout to prevent hanging if Ollama freezes
        )
        self.local_model = os.getenv("LOCAL_MODEL", "llama3.2:3b")

        # Determine if local self-critique should be enabled
        enable_critique_env = os.getenv("ENABLE_LOCAL_CRITIQUE")
        if enable_critique_env is not None:
            self.enable_local_critique = enable_critique_env.lower() in ("true", "1", "yes")
        else:
            # Auto-disable critique for smaller models (e.g. 1b or 3b) to prevent resource-heavy hallucinations
            model_lower = self.local_model.lower()
            if "1b" in model_lower or "3b" in model_lower:
                self.enable_local_critique = False
            else:
                self.enable_local_critique = True

        # Remote client configuration (Fireworks AI)
        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        fireworks_base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        self.remote_client = None
        if fireworks_api_key:
            self.remote_client = OpenAI(
                base_url=fireworks_base_url,
                api_key=fireworks_api_key
            )
        else:
            print("Warning: FIREWORKS_API_KEY is not set. Remote routing calls will fail.")
        
        # Read allowed models from environment (Track 1 requirement)
        allowed_models_env = os.getenv("ALLOWED_MODELS")
        if allowed_models_env:
            allowed_models = [m.strip() for m in allowed_models_env.split(",") if m.strip()]
            if allowed_models:
                # Use the first permitted model as our primary remote model
                self.remote_model = allowed_models[0]
            else:
                self.remote_model = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
        else:
            self.remote_model = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
        
        # Remote model pricing per 1M tokens (defaults to $0.90 per 1M tokens for Llama 3.1 70B)
        self.remote_price_per_1m_tokens = float(os.getenv("REMOTE_PRICE_PER_1M_TOKENS", "0.90"))

        # Detect AMD GPU availability
        self.gpu_info = self.detect_gpu()

    def detect_gpu(self) -> dict:
        """
        Detect AMD GPU availability and return hardware info.
        Gracefully degrades on non-AMD systems (Mac, Intel, etc.).
        Returns dict with keys: available, name, vram_gb, driver.
        """
        info = {
            "available": False,
            "name": "",
            "vram_gb": 0,
            "driver": "",
            "backend": "cpu"
        }

        # Method 1: rocminfo (most reliable for ROCm systems)
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                info["available"] = True
                info["backend"] = "rocm"
                # Parse GPU name from rocminfo output
                for line in result.stdout.split('\n'):
                    if "Marketing Name:" in line:
                        name = line.split(":", 1)[1].strip()
                        if name and "AMD" in name:
                            info["name"] = name
                    elif "Device Type:" in line and "GPU" in line:
                        pass  # Confirms GPU device
                # Try to get VRAM from rocminfo
                for line in result.stdout.split('\n'):
                    if "Pool Size:" in line:
                        match = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB)', line)
                        if match:
                            val = float(match.group(1))
                            unit = match.group(2)
                            info["vram_gb"] = int(val / 1024) if unit == "MB" else int(val)
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            pass  # rocminfo not available — not an AMD ROCm system

        # Method 2: Check via torch (ROCm PyTorch reports HIP via torch.cuda)
        if not info["available"]:
            try:
                import torch
                if torch.cuda.is_available() and hasattr(torch.version, 'hip'):
                    info["available"] = True
                    info["backend"] = "rocm"
                    info["name"] = torch.cuda.get_device_name(0)
                    try:
                        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                    except Exception:
                        pass
                    info["driver"] = f"ROCm (HIP) {torch.version.hip or 'unknown'}"
            except ImportError:
                pass  # PyTorch not installed

        # Method 3: Check via rocm-smi (alternative ROCm tool)
        if not info["available"]:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    info["available"] = True
                    info["backend"] = "rocm"
                    for line in result.stdout.split('\n'):
                        if line.strip() and "=" in line:
                            info["name"] = line.split("=")[-1].strip()
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return info

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the token count for a given text using tiktoken.
        Falls back to a character-based heuristic if tiktoken fails.
        """
        try:
            # Use cl100k_base encoding which is the default for most modern models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: ~4 characters per token (common heuristic)
            return len(text) // 4 + 1

    def call_local(self, prompt: str, system_prompt: str = None, temperature: float = 0.2, max_tokens: int = 1000, json_mode: bool = False) -> str:
        """
        Sends a query to the local model via Ollama.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json"}

        try:
            response = self.local_client.chat.completions.create(
                model=self.local_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error executing local query: {str(e)}"

    def call_remote(self, prompt: str, system_prompt: str = None, temperature: float = 0.2, max_tokens: int = 1000) -> tuple[str, int, int]:
        """
        Sends a query to the remote Fireworks API.
        Returns:
            tuple: (response_text, prompt_tokens, completion_tokens)
        """
        if not self.remote_client:
            raise ValueError("FIREWORKS_API_KEY is not configured in .env file.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.remote_client.chat.completions.create(
                model=self.remote_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            return response.choices[0].message.content, prompt_tokens, completion_tokens
        except Exception as e:
            return f"Error executing remote query: {str(e)}", 0, 0
