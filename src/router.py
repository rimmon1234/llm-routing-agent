import os
import json
import time
from .client import LLMClient
from .evaluator import ResponseEvaluator
from .style import S

# Compact repair prompt (no system prompt needed — the repair prompt is self-contained)
_REPAIR_TMPL = "Q:{q}\nA:{r}\nERR:{e}\nFix ({fmt}{schema}):"

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
                print(f"   {S.warn('[Cache]')} Warning: Failed to load cache file: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   {S.warn('[Cache]')} Warning: Failed to write cache file: {e}")

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

    def _run_local_with_retry(self, query: str, response_format: str, schema: any,
                               use_json_mode: bool, max_retries: int, result: dict) -> tuple:
        """Run local model with retry loop. Updates result in place. Returns (response, is_valid, error_reason)."""
        result["local_attempts"] += 1
        local_res = self.client.call_local(query, json_mode=use_json_mode)
        result["prompt_tokens_local"] += self.client.estimate_tokens(query)
        result["completion_tokens_local"] += self.client.estimate_tokens(local_res)

        is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)

        if not is_valid and max_retries > 0 and "Execution error" not in error_reason:
            print(f"   [Router] Repairing ({error_reason})...")
            schema_hint = f"|{schema}" if schema else ""
            for attempt in range(1, max_retries + 1):
                time.sleep(0.5)
                result["local_attempts"] += 1

                repair_prompt = _REPAIR_TMPL.format(
                    q=query, r=local_res, e=error_reason,
                    fmt=response_format, schema=schema_hint
                )
                result["prompt_tokens_local"] += self.client.estimate_tokens(repair_prompt)

                local_res = self.client.call_local(
                    prompt=repair_prompt,
                    temperature=0.1,
                    json_mode=use_json_mode
                )
                result["completion_tokens_local"] += self.client.estimate_tokens(local_res)

                is_valid, error_reason = self.evaluator.evaluate(query, local_res, response_format, schema)
                if is_valid:
                    print(f"   {S.good('[Router]')} Repaired attempt {S.good(str(attempt))}!")
                    break
                else:
                    print(f"   {S.warn('[Router]')} Attempt {attempt} failed ({S.warn(error_reason)}).")

        return local_res, is_valid, error_reason

    def _handle_remote_call(self, query: str, result: dict):
        """Call remote API, update result in place. Returns the response."""
        result["remote_attempts"] += 1
        res, pt, ct = self.client.call_remote(query)
        result["prompt_tokens_remote"] = pt
        result["completion_tokens_remote"] = ct
        result["response"] = res
        if pt == 0 and ct == 0 and self._is_error(res):
            return res  # caller checks _is_error
        total_tokens = pt + ct
        result["cost_dollars"] = (total_tokens / 1_000_000.0) * self.client.remote_price_per_1m_tokens
        return res

    def _compute_cost_saved(self, result: dict) -> float:
        """Proportion of total tokens handled locally."""
        local_tok = result["prompt_tokens_local"] + result["completion_tokens_local"]
        remote_tok = result["prompt_tokens_remote"] + result["completion_tokens_remote"]
        total = local_tok + remote_tok
        return local_tok / total if total > 0 else 0.0

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
                result = cached_result.copy()
                result.setdefault("prompt_tokens_local", 0)
                result.setdefault("completion_tokens_local", 0)
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
            "prompt_tokens_local": 0,
            "completion_tokens_local": 0,
            "prompt_tokens_remote": 0,
            "completion_tokens_remote": 0,
            "fallback_triggered": False,
            "cost_saved": 0.0,
            "latency_sec": 0.0,
            "cost_dollars": 0.0,
            "cached": False
        }

        use_json_mode = (response_format.lower() == "json")

        if strategy == "always_local":
            result["route_chosen"] = "local"
            result["local_attempts"] = 1
            result["response"] = self.client.call_local(query, json_mode=use_json_mode)
            result["prompt_tokens_local"] = self.client.estimate_tokens(query)
            result["completion_tokens_local"] = self.client.estimate_tokens(result["response"])
            result["cost_saved"] = 1.0

        elif strategy == "always_remote":
            result["route_chosen"] = "remote"
            self._handle_remote_call(query, result)
            result["cost_saved"] = 0.0
            if result["prompt_tokens_remote"] == 0 and result["completion_tokens_remote"] == 0 and self._is_error(result["response"]):
                result["route_chosen"] = "remote_failed"

        elif strategy == "fallback":
            local_res, is_valid, error_reason = self._run_local_with_retry(
                query, response_format, schema, use_json_mode, max_retries, result
            )
            if is_valid:
                result["route_chosen"] = "local"
                result["response"] = local_res
                result["cost_saved"] = self._compute_cost_saved(result)
            else:
                print(f"   {S.warn('[Router]')} Falling back...")
                result["fallback_triggered"] = True
                self._handle_remote_call(query, result)
                if result["prompt_tokens_remote"] == 0 and result["completion_tokens_remote"] == 0 and self._is_error(result["response"]):
                    print(f"   {S.error('[Router]')} Remote failed. Using local best-effort.")
                    result["route_chosen"] = "local_best_effort"
                    result["response"] = local_res
                    result["cost_saved"] = 1.0
                else:
                    result["route_chosen"] = "remote_fallback"
                    result["cost_saved"] = self._compute_cost_saved(result)

        elif strategy == "predictive":
            is_complex = self._predict_complexity(query, response_format)

            if is_complex:
                result["route_chosen"] = "remote"
                self._handle_remote_call(query, result)
                result["cost_saved"] = 0.0
            else:
                local_res, is_valid, error_reason = self._run_local_with_retry(
                    query, response_format, schema, use_json_mode, max_retries, result
                )
                if is_valid:
                    result["route_chosen"] = "local"
                    result["response"] = local_res
                    result["cost_saved"] = self._compute_cost_saved(result)
                else:
                    result["fallback_triggered"] = True
                    self._handle_remote_call(query, result)
                    if result["prompt_tokens_remote"] == 0 and result["completion_tokens_remote"] == 0 and self._is_error(result["response"]):
                        print(f"   {S.error('[Router]')} Remote failed. Using local best-effort.")
                        result["route_chosen"] = "local_best_effort"
                        result["response"] = local_res
                        result["cost_saved"] = 1.0
                    else:
                        result["route_chosen"] = "remote_fallback"
                        result["cost_saved"] = self._compute_cost_saved(result)

        result["latency_sec"] = time.perf_counter() - start_time

        # Save to cache if no errors and caching is allowed
        if not no_cache and result["response"]:
            if not result["response"].lower().startswith("error executing"):
                self.cache.set(query, strategy, response_format, schema_str, result)

        return result

    def _is_error(self, response: str) -> bool:
        """Check if a response string is an error message from the LLM client."""
        if not response:
            return True
        return response.lower().startswith("error executing")

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
