import os
import json
import time
from .client import LLMClient
from .evaluator import ResponseEvaluator

class RouterCache:
    def __init__(self, cache_file: str = ".router_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"   [Cache] Warning: Failed to load cache file: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   [Cache] Warning: Failed to write cache file: {e}")

    def _normalize_query(self, query: str) -> str:
        import re
        q = query.strip().lower()
        q = re.sub(r'\s+', ' ', q)
        return q

    def get(self, query: str, strategy: str, response_format: str, schema: str = None) -> dict:
        normalized_query = self._normalize_query(query)
        key = f"{normalized_query}||{strategy}||{response_format}||{schema or ''}"
        return self.cache.get(key)

    def set(self, query: str, strategy: str, response_format: str, schema: str, entry: dict):
        normalized_query = self._normalize_query(query)
        key = f"{normalized_query}||{strategy}||{response_format}||{schema or ''}"
        self.cache[key] = entry
        self._save_cache()

    def clear(self):
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                pass


class HybridRouter:
    def __init__(self, client: LLMClient = None, evaluator: ResponseEvaluator = None, cache_file: str = ".router_cache.json"):
        self.client = client or LLMClient()
        self.evaluator = evaluator or ResponseEvaluator(self.client)
        self.cache = RouterCache(cache_file)

    def route_and_execute(self, query: str, strategy: str = "fallback", response_format: str = "text", schema: any = None, max_retries: int = 2, no_cache: bool = False) -> dict:
        """
        Routes the query based on the strategy and returns execution logs and output.
        
        Strategies:
            'always_local': Direct execution via Ollama (No validation)
            'always_remote': Direct execution via Fireworks AI
            'fallback': Try local first, evaluate, fallback to remote if validation fails
            'predictive': Evaluate complexity upfront, then route directly to remote or run local-first
        """
        start_time = time.perf_counter()
        schema_str = str(schema) if schema is not None else None

        # Cap retries for small models (1b or 3b) to prevent local resource spikes
        model_lower = self.client.local_model.lower()
        if "1b" in model_lower or "3b" in model_lower:
            max_retries = min(max_retries, 1)

        if not no_cache:
            cached_result = self.cache.get(query, strategy, response_format, schema_str)
            if cached_result:
                # Return copy with cached flag set to True
                result = cached_result.copy()
                result["cached"] = True
                result["latency_sec"] = time.perf_counter() - start_time
                return result

        result = {
            "query": query,
            "strategy": strategy,
            "route_chosen": None,
            "response": None,
            "local_attempts": 0,
            "remote_attempts": 0,
            "prompt_tokens_remote": 0,
            "completion_tokens_remote": 0,
            "fallback_triggered": False,
            "cost_saved": 0.0,
            "latency_sec": 0.0,
            "cost_dollars": 0.0,
            "cached": False
        }

        # Check if local model expects strict JSON mode
        use_json_mode = (response_format.lower() == "json")

        if strategy == "always_local":
            result["route_chosen"] = "local"
            result["local_attempts"] = 1
            result["response"] = self.client.call_local(query, json_mode=use_json_mode)
            result["cost_saved"] = 1.0
            
        elif strategy == "always_remote":
            result["route_chosen"] = "remote"
            result["remote_attempts"] = 1
            res, pt, ct = self.client.call_remote(query)
            result["response"] = res
            result["prompt_tokens_remote"] = pt
            result["completion_tokens_remote"] = ct
            result["cost_saved"] = 0.0
            total_tokens = pt + ct
            result["cost_dollars"] = (total_tokens / 1_000_000.0) * self.client.remote_price_per_1m_tokens
            
        elif strategy == "fallback":
            # Step 1: Run local model
            result["local_attempts"] += 1
            local_res = self.client.call_local(query, json_mode=use_json_mode)
            
            # Step 2: Validate local response
            is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)
            
            # Local Self-Correction (Retry Loop)
            if not is_valid and max_retries > 0 and "Execution error" not in error_reason:
                print(f"   [Router] Local validation failed ({error_reason}). Initiating local repair loop...")
                for attempt in range(1, max_retries + 1):
                    time.sleep(0.5)  # Cooldown delay to prevent system overload
                    result["local_attempts"] += 1
                    
                    # Construct repair instructions
                    system_prompt = (
                        "You are a helpful assistant. Correct the previous response based on the validation error. "
                        "Do not explain the error; just output the corrected, compliant response."
                    )
                    schema_info = f" conforming to schema/requirements: {schema}" if schema else ""
                    repair_prompt = (
                        f"Original Query: {query}\n\n"
                        f"Previous Response: {local_res}\n\n"
                        f"Validation Error: {error_reason}\n\n"
                        f"Corrected Response (Must be in {response_format} format{schema_info}):"
                    )
                    
                    local_res = self.client.call_local(
                        prompt=repair_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,  # Lower temp to increase deterministic correctness
                        json_mode=use_json_mode
                    )
                    
                    # Evaluate again
                    is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)
                    if is_valid:
                        print(f"   [Router] Local repair successful on attempt {attempt}!")
                        break
                    else:
                        print(f"   [Router] Local repair attempt {attempt} failed ({error_reason}).")

            if is_valid:
                result["route_chosen"] = "local"
                result["response"] = local_res
                result["cost_saved"] = 1.0
            else:
                # Step 3: Local failed all attempts, fallback to remote
                print("   [Router] Local repair failed or unavailable. Falling back to remote...")
                result["fallback_triggered"] = True
                result["route_chosen"] = "remote_fallback"
                result["remote_attempts"] += 1
                res, pt, ct = self.client.call_remote(query)
                result["response"] = res
                result["prompt_tokens_remote"] = pt
                result["completion_tokens_remote"] = ct
                result["cost_saved"] = 0.0
                total_tokens = pt + ct
                result["cost_dollars"] = (total_tokens / 1_000_000.0) * self.client.remote_price_per_1m_tokens

        elif strategy == "predictive":
            # Step 1: Predict complexity upfront
            is_complex = self._predict_complexity(query, response_format)
            
            if is_complex:
                # Route directly to remote
                result["route_chosen"] = "remote"
                result["remote_attempts"] += 1
                res, pt, ct = self.client.call_remote(query)
                result["response"] = res
                result["prompt_tokens_remote"] = pt
                result["completion_tokens_remote"] = ct
                result["cost_saved"] = 0.0
                total_tokens = pt + ct
                result["cost_dollars"] = (total_tokens / 1_000_000.0) * self.client.remote_price_per_1m_tokens
            else:
                # Run local first
                result["local_attempts"] += 1
                local_res = self.client.call_local(query, json_mode=use_json_mode)
                is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)
                
                # Local Self-Correction (Retry Loop)
                if not is_valid and max_retries > 0 and "Execution error" not in error_reason:
                    print(f"   [Router] Local validation failed ({error_reason}). Initiating local repair loop...")
                    for attempt in range(1, max_retries + 1):
                        time.sleep(0.5)  # Cooldown delay to prevent system overload
                        result["local_attempts"] += 1
                        
                        system_prompt = (
                            "You are a helpful assistant. Correct the previous response based on the validation error. "
                            "Do not explain the error; just output the corrected, compliant response."
                        )
                        schema_info = f" conforming to schema/requirements: {schema}" if schema else ""
                        repair_prompt = (
                            f"Original Query: {query}\n\n"
                            f"Previous Response: {local_res}\n\n"
                            f"Validation Error: {error_reason}\n\n"
                            f"Corrected Response (Must be in {response_format} format{schema_info}):"
                        )
                        
                        local_res = self.client.call_local(
                            prompt=repair_prompt,
                            system_prompt=system_prompt,
                            temperature=0.1,
                            json_mode=use_json_mode
                        )
                        is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)
                        if is_valid:
                            print(f"   [Router] Local repair successful on attempt {attempt}!")
                            break
                        else:
                            print(f"   [Router] Local repair attempt {attempt} failed ({error_reason}).")

                if is_valid:
                    result["route_chosen"] = "local"
                    result["response"] = local_res
                    result["cost_saved"] = 1.0
                else:
                    result["fallback_triggered"] = True
                    result["route_chosen"] = "remote_fallback"
                    result["remote_attempts"] += 1
                    res, pt, ct = self.client.call_remote(query)
                    result["response"] = res
                    result["prompt_tokens_remote"] = pt
                    result["completion_tokens_remote"] = ct
                    result["cost_saved"] = 0.0
                    total_tokens = pt + ct
                    result["cost_dollars"] = (total_tokens / 1_000_000.0) * self.client.remote_price_per_1m_tokens

        # Record total latency
        result["latency_sec"] = time.perf_counter() - start_time

        # Save to cache if no errors and caching is allowed
        if not no_cache and result["response"] and "error executing" not in result["response"].lower():
            self.cache.set(query, strategy, response_format, schema_str, result)

        return result

    def _predict_complexity(self, query: str, response_format: str = "text") -> bool:
        """
        Predicts query complexity using fast, rule-based heuristics to avoid
        local LLM classification overhead (saving CPU/GPU resources).
        """
        # If format is python, route to remote immediately as small local models are not reliable for coding
        if response_format.lower() == "python":
            return True

        import re
        query_lower = query.lower()
        
        # 1. Check for programming/coding code blocks or tags
        if "```" in query:
            return True
            
        # 2. Check for long queries (typically require detailed analysis/synthesis)
        word_count = len(query_lower.split())
        if word_count > 80:
            return True
            
        # 3. Check for coding/architecture indicators with word boundaries
        complex_words = [
            "implement", "optimize", "optimise", "analyze", "analyse", "prove",
            "debug", "error", "exception", "refactor", "algorithm", "recursive", "recursion",
            "compile", "dependency", "api", "database", "sql", "regex", "schema"
        ]
        # Match word boundaries for indicators
        for word in complex_words:
            if re.search(r'\b' + re.escape(word) + r'\b', query_lower):
                return True
                
        # Substring phrases that indicate complexity
        complex_phrases = [
            "explain the difference", "write a python", "complexity of", "architecture"
        ]
        if any(phrase in query_lower for phrase in complex_phrases):
            return True
            
        # 4. Check for reasoning, math, and logic indicators with word boundaries
        reasoning_words = [
            "solve", "calculate", "equation", "formula", "logic", "proof", "prove"
        ]
        for word in reasoning_words:
            if re.search(r'\b' + re.escape(word) + r'\b', query_lower):
                return True
                
        reasoning_phrases = [
            "step-by-step", "step by step", "compare and contrast", "why does", "how do i"
        ]
        if any(phrase in query_lower for phrase in reasoning_phrases):
            return True
            
        return False
