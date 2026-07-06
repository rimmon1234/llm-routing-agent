import os
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
        self.remote_client = None
        if fireworks_api_key:
            self.remote_client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=fireworks_api_key
            )
        else:
            print("Warning: FIREWORKS_API_KEY is not set. Remote routing calls will fail.")
        self.remote_model = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
        
        # Remote model pricing per 1M tokens (defaults to $0.90 per 1M tokens for Llama 3.1 70B)
        self.remote_price_per_1m_tokens = float(os.getenv("REMOTE_PRICE_PER_1M_TOKENS", "0.90"))

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
