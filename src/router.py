import os
import json
import time
import re
from dataclasses import dataclass, field
from .client import LLMClient
from .evaluator import ResponseEvaluator
from .style import S

# Compact repair prompt (no system prompt needed — the repair prompt is self-contained)
_REPAIR_TMPL = "Q:{q}\nA:{r}\nERR:{e}\nFix ({fmt}{schema}):"

@dataclass
class ComplexityReport:
    score: float
    normalized_score: float
    risk_score: float
    confidence: float
    decision: str  # "local" or "remote"
    routing_zone: str  # "Definitely Local", "Borderline", "Definitely Remote"
    matched_features: list[str] = field(default_factory=list)
    matched_domains: list[str] = field(default_factory=list)
    estimated_output_size: str = "medium"  # "low", "medium", "high"
    estimated_task_count: int = 1
    comparison_dimension_count: int = 0
    reasoning_depth: int = 0
    expected_failure_modes: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    comp_entities_count: int = 0
    comparison_count: int = 0

@dataclass
class RoutingFeatures:
    task_count: int
    constraint_count: int
    entities_count: int
    tech_depth: int
    output_size: str
    ambiguity_count: int
    reasoning_depth: int
    math_matches: int
    matched_domains: list[str]
    matched_entities: list[str]
    comp_dimension_count: int
    comp_entities_count: int
    comparison_count: int

class PredictiveRouterConfig:
    # 1. THRESHOLDS & ROUTING REGIMES
    ZONE_LOCAL_LIMIT = 0.30       # Definitely Local boundary
    ZONE_REMOTE_LIMIT = 0.55      # Definitely Remote boundary
    RISK_THRESHOLD = 0.35         # Risk tolerance boundary for Borderline zone

    # 2. COMPLEXITY FEATURE WEIGHTS
    COMPLEXITY_WEIGHTS = {
        "task_count": 1.5,
        "constraint": 1.0,
        "entity": 1.2,
        "tech_depth": 2.0,
        "size_high": 2.5,
        "size_medium": 1.0,
        "ambiguity": 1.5,
        "length_medium": 1.5,
        "length_long": 3.0,
        "diversity": 1.0,
        "math_symbol": 2.5,
        "reasoning": 2.5,
        "domain_breadth": 2.0
    }

    # 3. RISK FEATURE WEIGHTS (Likelihood of local model failure)
    RISK_WEIGHTS = {
        "math_reasoning": 4.5,
        "code_generation": 4.0,
        "code_debugging": 3.5,
        "strict_constraints": 3.0,
        "distributed_systems": 4.5,
        "format_json": 3.5,
        "multi_domain": 2.5
    }

    # 4. TECH DOMAINS & KEYWORDS
    DOMAIN_KEYWORDS = {
        "Programming": {"python", "java", "c++", "cpp", "javascript", "js", "typescript", "ts", "rust", "compilation", "ast", "recursion", "multithreading", "garbage collection", "string", "function", "method", "class", "code", "program", "loop", "array", "list", "variable", "reverse"},
        "Databases": {"sql", "postgres", "mysql", "indexing", "normalization", "nosql", "sharding", "acid", "transaction", "isolation level", "query planner", "vector database"},
        "Distributed Systems": {"consensus", "raft", "paxos", "sharding", "replication", "consistency", "availability", "distributed", "partition", "split brain", "leader election", "consumer groups"},
        "Networking": {"tcp", "udp", "http", "dns", "latency", "bandwidth", "packet", "routing", "cidr", "socket", "tls", "ssl"},
        "Security": {"encryption", "decryption", "cryptography", "ssl", "tls", "auth", "jwt", "xss", "csrf", "vulnerability", "buffer overflow", "access token", "authorization code"},
        "Artificial Intelligence": {"gradient descent", "neural network", "transformer", "backpropagation", "supervised", "unsupervised", "dataset", "llm", "embeddings"},
        "Cloud": {"kubernetes", "k8s", "docker", "container", "microservices", "aws", "azure", "gcp", "serverless"},
        "Operating Systems": {"kernel", "process", "thread", "mutex", "deadlock", "virtual memory", "scheduling", "sys call", "paging"},
        "Mathematics": {"algebra", "calculus", "derivative", "integral", "matrix", "modulo", "equation", "probability", "statistics", "combinatorics", "logic", "true", "false", "boolean", "proof", "prove", "if"},
        "Finance": {"portfolio", "projection", "depreciation", "interest", "derivative", "yield", "amortization", "compound"},
        "Science": {"physics", "chemistry", "biology", "molecule", "reaction", "formula", "velocity", "gravity"},
        "Infrastructure": {"load balancer", "reverse proxy", "cdn", "dns", "firewall", "gateway", "service mesh"}
    }

    # 5. FEATURE KEYWORDS AND MATCHERS
    FEATURE_KEYWORDS = {
        "task_verbs": {"compare", "explain", "design", "implement", "write", "generate", "review", "summarize", "translate", "recommend", "justify", "evaluate", "classify", "extract", "debug", "fix", "analyze"},
        "constraints": {"exactly", "at least", "at most", "strictly", "must", "without", "using only", "json", "markdown", "bullet points", "table", "schema", "format", "less than", "more than"},
        "size_low": {"yes or no", "single word", "one word", "binary", "true or false", "tl;dr"},
        "size_high": {"detailed", "comprehensive", "step-by-step", "complete", "tutorial", "guide", "architecture", "deep dive", "full implementation", "in-depth", "elaborate", "long format", "explanations", "include", "cover", "describe", "discuss", "walkthrough", "for each", "recommend", "justify", "provide examples"},
        "ambiguity": {"ethical", "economic", "legal", "technical", "historical", "advantages", "disadvantages", "tradeoffs", "pros and cons", "pros & cons", "alternative"},
        "reasoning": {"why", "justify", "prove", "derive", "evaluate", "critique", "infer", "predict", "analyze", "reason", "deduce", "tradeoffs", "pros and cons", "advantages", "disadvantages"},
        "comparison_phrases": {"in terms of", "with respect to", "according to", "considering", "based on", "across"},
        "comparison_metrics": {"performance", "latency", "throughput", "scalability", "availability", "durability", "consistency", "security", "maintainability", "cost", "complexity"},
        "popular_entities": {"kafka", "rabbitmq", "redis", "docker", "kubernetes", "postgres", "spark", "hadoop", "nginx", "aws", "gcp", "azure", "ollama", "fireworks"}
    }


    # 6. EXPECTED FAILURE MODES Mapping
    FAILURE_MODES = {
        "math_reasoning": "Mathematical reasoning error / arithmetic hallucination",
        "code_generation": "Code generation syntax error or logical bug",
        "code_debugging": "Failure to identify hidden bugs or logical flaws",
        "strict_constraints": "Format constraint violation (length, sentence count)",
        "distributed_systems": "Distributed systems reasoning flaw / concurrency deadlock",
        "format_json": "Invalid JSON structure or schema validation crash",
        "multi_domain": "Context drift / multi-domain hallucination"
    }

    # 7. MATH SYMBOLS
    MATH_SYMBOLS = {'+', '-', '*', '/', '=', '^', '<', '>', 'sqrt', 'log', 'sum', 'integral', 'derivative'}


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
        self._precompile_patterns()

    def _precompile_patterns(self) -> None:
        """Precompile and cache regexes and lookup structures to optimize matching performance to O(N)."""
        # Precompile domains
        self._domain_word_sets = {}
        self._domain_phrase_patterns = {}
        self._domain_phrases_list = {}
        for domain, keywords in PredictiveRouterConfig.DOMAIN_KEYWORDS.items():
            word_set = set()
            phrases = []
            for kw in keywords:
                if ' ' in kw or '-' in kw:
                    phrases.append(kw)
                else:
                    word_set.add(kw)
            self._domain_word_sets[domain] = word_set
            self._domain_phrases_list[domain] = phrases
            if phrases:
                escaped = [re.escape(p) for p in phrases]
                self._domain_phrase_patterns[domain] = re.compile(r'\b(?:' + '|'.join(escaped) + r')\b')

        # Precompile comparison phrase pattern
        comp_phrases = [re.escape(p) for p in PredictiveRouterConfig.FEATURE_KEYWORDS["comparison_phrases"]]
        self._comp_phrases_pattern = re.compile(r'\b(?:' + '|'.join(comp_phrases) + r')\b')
        
        # Precompile comparison metrics pattern
        comp_metrics = [re.escape(m) for m in PredictiveRouterConfig.FEATURE_KEYWORDS["comparison_metrics"]]
        self._comp_metrics_pattern = re.compile(r'\b(?:' + '|'.join(comp_metrics) + r')\b')

        # Precompile output size patterns
        size_low_esc = [re.escape(k) for k in PredictiveRouterConfig.FEATURE_KEYWORDS["size_low"]]
        self._size_low_pattern = re.compile(r'\b(?:' + '|'.join(size_low_esc) + r')\b')
        
        size_high_esc = [re.escape(k) for k in PredictiveRouterConfig.FEATURE_KEYWORDS["size_high"]]
        self._size_high_pattern = re.compile(r'\b(?:' + '|'.join(size_high_esc) + r')\b')

        # Precompile constraints pattern
        constraints_esc = [re.escape(c) for c in PredictiveRouterConfig.FEATURE_KEYWORDS["constraints"]]
        self._constraints_pattern = re.compile(r'\b(?:' + '|'.join(constraints_esc) + r')\b')

        # Precompile ambiguity pattern
        ambiguity_esc = [re.escape(a) for a in PredictiveRouterConfig.FEATURE_KEYWORDS["ambiguity"]]
        self._ambiguity_pattern = re.compile(r'\b(?:' + '|'.join(ambiguity_esc) + r')\b')


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

    def _get_task_count(self, tokens_set: set) -> int:
        """Estimate the count of requested actions in the query using set intersection."""
        return len(tokens_set.intersection(PredictiveRouterConfig.FEATURE_KEYWORDS["task_verbs"]))

    def _get_constraint_count(self, query_lower: str) -> int:
        """Count custom format or strict constraints in the query using precompiled patterns."""
        return len(set(self._constraints_pattern.findall(query_lower)))

    def _get_entities(self, query: str, query_lower: str) -> tuple[int, list[str]]:
        """Detect referenced technology stacks and proper noun entities using word boundaries and proper noun filters."""
        # Respect word boundaries using set intersection of words
        query_words = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', query_lower))
        matched_popular = query_words.intersection(PredictiveRouterConfig.FEATURE_KEYWORDS["popular_entities"])
        
        # Parse sentences/clauses to identify start words to avoid false positive proper nouns
        clauses = re.split(r'[.!?\n]+', query)
        start_words = set()
        for clause in clauses:
            clause_clean = clause.strip()
            if clause_clean:
                words = clause_clean.split()
                if words:
                    first_word = words[0].strip('",\'()[]{}*:;,-')
                    start_words.add(first_word.lower())
                    
        # Extract capitalized words
        capitalized_words = set(re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\b', query))
        
        # Stop words and common sentence starters (case-insensitive check)
        stop_words = {
            "i", "a", "the", "we", "you", "they", "he", "she", "it", "this", "that", "these", "those",
            "how", "what", "why", "when", "where", "who", "which", "whose", "whom",
            "write", "create", "explain", "compare", "describe", "design", "implement", "summarize",
            "calculate", "evaluate", "list", "extract", "find", "show", "give", "please", "can", "could",
            "would", "should", "will", "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "if", "then", "else", "or", "and", "but", "not", "so", "for", "with"
        }
        
        filtered_proper_nouns = set()
        for word in capitalized_words:
            word_lower = word.lower()
            if word_lower in stop_words:
                continue
            if word_lower in start_words and word_lower not in PredictiveRouterConfig.FEATURE_KEYWORDS["popular_entities"]:
                continue
            filtered_proper_nouns.add(word)
            
        all_entities = matched_popular.union({w.lower() for w in filtered_proper_nouns})
        entities_list = sorted(list(all_entities))
        return len(entities_list), entities_list

    def _get_entities_count(self, query: str, query_lower: str) -> int:
        """Detect referenced technology stacks and proper noun entities case-insensitively (backward compatibility interface)."""
        count, _ = self._get_entities(query, query_lower)
        return count

    def _get_tech_depth_count(self, query_lower: str) -> int:
        """Count key technical terms appearing across all registered domains in O(N) time."""
        words = set(re.findall(r'\b\w+\b', query_lower))
        count = 0
        for domain in PredictiveRouterConfig.DOMAIN_KEYWORDS:
            # Intersect with words in query to match whole words
            word_set = self._domain_word_sets.get(domain, set())
            count += len(words.intersection(word_set))
            
            # Count multi-word keywords
            pattern = self._domain_phrase_patterns.get(domain)
            if pattern:
                count += len(pattern.findall(query_lower))
        return count

    def _estimate_output_size(self, query_lower: str, word_count: int) -> str:
        """Predict output length based on deliverables and query length using precompiled patterns."""
        if word_count < 15 and self._size_low_pattern.search(query_lower):
            return "LOW"
        if word_count > 120 or self._size_high_pattern.search(query_lower):
            return "HIGH"
        return "MEDIUM"

    def _get_ambiguity_count(self, query_lower: str) -> int:
        """Count markers representing multiple perspectives or tradeoffs."""
        return len(set(self._ambiguity_pattern.findall(query_lower)))

    def _get_reasoning_depth(self, query_lower: str) -> int:
        """Count explicit reasoning request indicators using word-boundary matching."""
        words = set(re.findall(r'\b\w+\b', query_lower))
        depth = 0
        
        for r in PredictiveRouterConfig.FEATURE_KEYWORDS["reasoning"]:
            if ' ' in r or '-' in r:
                if r in query_lower:
                    depth += 1
            else:
                if r in words:
                    depth += 1
                    
        if words.intersection({"true", "false", "logic"}):
            depth += 1
            
        return depth

    def _get_math_matches(self, query_lower: str) -> int:
        """Count occurrences of mathematical symbols and terms."""
        count = 0
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        for sym in PredictiveRouterConfig.MATH_SYMBOLS:
            if len(sym) == 1:
                if sym in query_lower:
                    count += 1
            else:
                if sym in words:
                    count += 1
                    
        math_terms = {"calculate", "equation", "formula", "derivative", "integral", "matrix", "modulo", "solve"}
        for term in math_terms:
            if term in words:
                count += 1
                
        return count

    def _get_matched_domains(self, query_lower: str) -> list[str]:
        """Collect all matching technical domain classifications in O(N) time."""
        words = set(re.findall(r'\b\w+\b', query_lower))
        matched = []
        for domain in PredictiveRouterConfig.DOMAIN_KEYWORDS:
            # Check single word keywords via O(1) set intersection
            word_set = self._domain_word_sets.get(domain, set())
            if words.intersection(word_set):
                matched.append(domain)
                continue
            
            # Check multi-word phrases via precompiled regex pattern
            pattern = self._domain_phrase_patterns.get(domain)
            if pattern and pattern.search(query_lower):
                matched.append(domain)
        return matched

    def _get_comparison_dimension_count(self, query_lower: str) -> int:
        """Count comparison dimensions using precompiled word-boundary regexes."""
        if self._comp_phrases_pattern.search(query_lower):
            matched_metrics = set(self._comp_metrics_pattern.findall(query_lower))
            return len(matched_metrics)
        return 0

    def _get_comparison_features(self, query_lower: str, entities_list: list[str]) -> tuple[int, int, int]:
        """Evaluate comparison metrics and count comparison dimensions and comparison entities."""
        has_phrase = (
            self._comp_phrases_pattern.search(query_lower) is not None or 
            any(v in query_lower for v in [" vs ", " versus ", "compare"])
        )
        if has_phrase:
            matched_metrics = set(self._comp_metrics_pattern.findall(query_lower))
            comp_dim_count = len(matched_metrics)
            comp_ent_count = len(entities_list)
            comparison_count = comp_dim_count + comp_ent_count
            return comp_dim_count, comp_ent_count, comparison_count
        return 0, 0, 0

    def extract_routing_features(self, query: str, response_format: str = "text") -> RoutingFeatures:
        """Orchestrate the O(N) extraction of all complexity and risk features."""
        query_lower = query.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', query_lower)
        tokens = clean_text.split()
        tokens_set = set(tokens)
        word_count = len(tokens)
        
        task_count = self._get_task_count(tokens_set)
        constraint_count = self._get_constraint_count(query_lower)
        entities_count, matched_entities = self._get_entities(query, query_lower)
        tech_depth = self._get_tech_depth_count(query_lower)
        output_size = self._estimate_output_size(query_lower, word_count)
        ambiguity_count = self._get_ambiguity_count(query_lower)
        reasoning_depth = self._get_reasoning_depth(query_lower)
        math_matches = self._get_math_matches(query_lower)
        matched_domains = self._get_matched_domains(query_lower)
        
        comp_dim_count, comp_ent_count, comp_count = self._get_comparison_features(query_lower, matched_entities)
        
        return RoutingFeatures(
            task_count=task_count,
            constraint_count=constraint_count,
            entities_count=entities_count,
            tech_depth=tech_depth,
            output_size=output_size,
            ambiguity_count=ambiguity_count,
            reasoning_depth=reasoning_depth,
            math_matches=math_matches,
            matched_domains=matched_domains,
            matched_entities=matched_entities,
            comp_dimension_count=comp_dim_count,
            comp_entities_count=comp_ent_count,
            comparison_count=comp_count
        )


    def _has_non_python_language(self, query_lower: str) -> bool:
        """Determine if the query references popular non-Python programming languages."""
        non_python_langs = {"java", "javascript", "js", "cpp", "c++", "typescript", "ts", "rust"}
        return any(lang in query_lower for lang in non_python_langs)

    def _is_programming_debugging(self, query_lower: str, matched_domains: list[str]) -> bool:
        """Check if the task requires programming debugging or fixing bugs."""
        return "Programming" in matched_domains and any(kw in query_lower for kw in ("debug", "fix"))

    def _has_distributed_systems_indicators(self, query_lower: str, matched_domains: list[str]) -> bool:
        """Detect distributed systems keywords or domain matches."""
        distributed_risk_keywords = {"consensus", "raft", "paxos", "concurrency", "deadlock", "mutex", "split brain", "replication"}
        return any(kw in query_lower for kw in distributed_risk_keywords) or "Distributed Systems" in matched_domains

    def _has_large_document(self, query_lower: str) -> bool:
        """Identify indicators of large document context length or size constraints."""
        large_text_match = re.search(r'\b(\d{3,})\s*(?:-|\s+)(?:word|page|line)\b', query_lower)
        has_large_doc = any(phrase in query_lower for phrase in ["research paper", "scientific paper", "long article", "book chapter"])
        if large_text_match:
            try:
                val = int(large_text_match.group(1))
                if val > 500:
                    return True
            except ValueError:
                pass
        return has_large_doc

    def _calculate_risk_score(self, query_lower: str, response_format: str, math_matches: int, matched_domains: list[str]) -> float:
        """Assess the probability of local model output failure (raw score sum)."""
        risk_score = 0.0
        
        # Risk: JSON output requirement
        if response_format.lower() == "json" or "json" in query_lower:
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["format_json"]
            
        # Risk: Complex math arithmetic
        if math_matches > 0 or "Mathematics" in matched_domains:
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["math_reasoning"]
            
        # Risk: Code generation (especially non-Python compile targets)
        if self._has_non_python_language(query_lower):
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["code_generation"]
        elif "Programming" in matched_domains:
            if self._is_programming_debugging(query_lower, matched_domains):
                risk_score += PredictiveRouterConfig.RISK_WEIGHTS["code_debugging"]
            else:
                risk_score += 2.0  # Python/generic code gen
                
        # Risk: Distributed consensus concurrency conflicts
        if self._has_distributed_systems_indicators(query_lower, matched_domains):
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["distributed_systems"]
            
        # Risk: Crossover domains context drift
        if len(matched_domains) > 2:
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["multi_domain"]
            
        # Risk: Constraint density or large document reference
        constraint_count = self._get_constraint_count(query_lower)
        if constraint_count > 2 or self._has_large_document(query_lower):
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["strict_constraints"]
            
        # Risk: Logical reasoning puzzle
        if "logic" in query_lower or ("true" in query_lower and "false" in query_lower):
            risk_score += PredictiveRouterConfig.RISK_WEIGHTS["math_reasoning"]
            
        return risk_score

    def _get_expected_failure_modes(self, query_lower: str, response_format: str, math_matches: int, matched_domains: list[str]) -> list[str]:
        """Detail estimated failure types based on triggered risk indices."""
        modes = []
        if response_format.lower() == "json" or "json" in query_lower:
            modes.append(PredictiveRouterConfig.FAILURE_MODES["format_json"])
        if math_matches > 0 or "Mathematics" in matched_domains or "logic" in query_lower or ("true" in query_lower and "false" in query_lower):
            modes.append(PredictiveRouterConfig.FAILURE_MODES["math_reasoning"])
        if "Programming" in matched_domains:
            if self._is_programming_debugging(query_lower, matched_domains):
                modes.append(PredictiveRouterConfig.FAILURE_MODES["code_debugging"])
            else:
                modes.append(PredictiveRouterConfig.FAILURE_MODES["code_generation"])
        if "Distributed Systems" in matched_domains:
            modes.append(PredictiveRouterConfig.FAILURE_MODES["distributed_systems"])
        if len(matched_domains) > 2:
            modes.append(PredictiveRouterConfig.FAILURE_MODES["multi_domain"])
            
        # Constraints or large doc
        constraint_count = self._get_constraint_count(query_lower)
        if constraint_count > 2 or self._has_large_document(query_lower):
            modes.append(PredictiveRouterConfig.FAILURE_MODES["strict_constraints"])
            
        return modes

    def _evaluate_task_count(self, task_count: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from multi-task crossover queries."""
        score = 0.0
        features = []
        if task_count > 1:
            score += (task_count - 1) * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["task_count"]
            features.append(f"task_count_crossover({task_count})")
        return score, features

    def _evaluate_constraints(self, constraint_count: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from query constraint counts."""
        score = 0.0
        features = []
        if constraint_count > 0:
            score += constraint_count * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["constraint"]
            features.append(f"constraints({constraint_count})")
        return score, features

    def _evaluate_entities(self, entities_count: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from multiple proper noun entities and technology stacks."""
        score = 0.0
        features = []
        if entities_count > 1:
            score += (entities_count - 1) * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["entity"]
            features.append(f"multi_entity({entities_count})")
        return score, features

    def _evaluate_tech_depth(self, tech_depth: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from overall technical terminology depth."""
        score = 0.0
        features = []
        if tech_depth > 0:
            score += tech_depth * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["tech_depth"]
            features.append(f"tech_depth({tech_depth})")
        return score, features

    def _evaluate_output_size(self, output_size: str) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from expected output size predictions."""
        score = 0.0
        features = []
        if output_size == "HIGH":
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["size_high"]
            features.append("output_size_high")
        elif output_size == "MEDIUM":
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["size_medium"]
            features.append("output_size_medium")
        return score, features

    def _evaluate_ambiguity(self, ambiguity_count: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from open-endedness or ambiguity markers."""
        score = 0.0
        features = []
        if ambiguity_count > 0:
            score += ambiguity_count * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["ambiguity"]
            features.append(f"ambiguity({ambiguity_count})")
        return score, features

    def _evaluate_reasoning(self, reasoning_depth: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from deep reasoning prompts."""
        score = 0.0
        features = []
        if reasoning_depth > 0:
            score += reasoning_depth * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["reasoning"]
            features.append(f"reasoning_depth({reasoning_depth})")
        return score, features

    def _evaluate_math_references(self, math_matches: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from occurrences of mathematical symbols and terms."""
        score = 0.0
        features = []
        if math_matches > 0:
            score += math_matches * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["math_symbol"]
            features.append(f"math_references({math_matches})")
        return score, features

    def _evaluate_domain_breadth(self, matched_domains: list[str]) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from crossover of multiple knowledge domains."""
        score = 0.0
        features = []
        if len(matched_domains) > 1:
            score += (len(matched_domains) - 1) * PredictiveRouterConfig.COMPLEXITY_WEIGHTS["domain_breadth"]
            features.append(f"domain_breadth({len(matched_domains)})")
        return score, features

    def _evaluate_length_limits(self, word_count: int) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from query word limits."""
        score = 0.0
        features = []
        if word_count > 100:
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["length_long"]
            features.append("length_long")
        elif word_count > 50:
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["length_medium"]
            features.append("length_medium")
        return score, features

    def _evaluate_lexical_diversity(self, word_count: int, tokens_set: set[str]) -> tuple[float, list[str]]:
        """Evaluate complexity contribution from high unique-to-total word ratio (lexical diversity)."""
        score = 0.0
        features = []
        if word_count > 20:
            diversity = len(tokens_set) / word_count
            if diversity > 0.8:
                score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["diversity"]
                features.append("lexical_diversity_high")
        return score, features

    def _evaluate_complexity_math(self, math_matches: int, matched_domains: list[str]) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from math or math domain markers."""
        score = 0.0
        features = []
        if math_matches > 0 or "Mathematics" in matched_domains:
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["math_symbol"]
            features.append("complexity_math")
        return score, features

    def _evaluate_complexity_non_python(self, query_lower: str) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from non-Python target language coding query."""
        score = 0.0
        features = []
        if self._has_non_python_language(query_lower):
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["tech_depth"] * 1.5
            features.append("complexity_non_python")
        return score, features

    def _evaluate_complexity_debugging(self, query_lower: str, matched_domains: list[str]) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from programming debugging or fixing code queries."""
        score = 0.0
        features = []
        if self._is_programming_debugging(query_lower, matched_domains):
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["tech_depth"]
            features.append("complexity_debugging")
        return score, features

    def _evaluate_complexity_distributed_systems(self, query_lower: str, matched_domains: list[str]) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from distributed systems consensus or concurrency queries."""
        score = 0.0
        features = []
        if self._has_distributed_systems_indicators(query_lower, matched_domains):
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["domain_breadth"] * 1.5
            features.append("complexity_distributed_systems")
        return score, features

    def _evaluate_complexity_large_document(self, query_lower: str) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from large documents or papers queries."""
        score = 0.0
        features = []
        if self._has_large_document(query_lower):
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["length_long"]
            features.append("complexity_large_document")
        return score, features

    def analyze_complexity(self, query: str, response_format: str = "text") -> ComplexityReport:
        """Performs full heuristic evaluation and feature extraction to classify query complexity and risk."""
        query_lower = query.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', query_lower)
        tokens = clean_text.split()
        tokens_set = set(tokens)
        word_count = len(tokens)
        
        # 1. Intermediate O(N) extraction to RoutingFeatures
        features = self.extract_routing_features(query, response_format)
        
        # 2. Score mapping
        score = 0.0
        matched_features = []
        
        # Sequentially evaluate all mapped complexity and risk-based features using routing features
        evaluators = [
            self._evaluate_task_count(features.task_count),
            self._evaluate_constraints(features.constraint_count),
            self._evaluate_entities(features.entities_count),
            self._evaluate_tech_depth(features.tech_depth),
            self._evaluate_output_size(features.output_size),
            self._evaluate_ambiguity(features.ambiguity_count),
            self._evaluate_reasoning(features.reasoning_depth),
            self._evaluate_math_references(features.math_matches),
            self._evaluate_domain_breadth(features.matched_domains),
            self._evaluate_length_limits(word_count),
            self._evaluate_lexical_diversity(word_count, tokens_set),
            self._evaluate_complexity_math(features.math_matches, features.matched_domains),
            self._evaluate_complexity_non_python(query_lower),
            self._evaluate_complexity_debugging(query_lower, features.matched_domains),
            self._evaluate_complexity_distributed_systems(query_lower, features.matched_domains),
            self._evaluate_complexity_large_document(query_lower)
        ]
        
        for feat_score, feat_names in evaluators:
            score += feat_score
            matched_features.extend(feat_names)
            
        # 3. Dynamic Normalization boundary
        max_possible = sum(PredictiveRouterConfig.COMPLEXITY_WEIGHTS.values())
        normalized_score = min(score / max_possible, 1.0)
        
        # 4. Risk estimation (normalized against highest possible single category weight)
        risk_score_raw = self._calculate_risk_score(query_lower, response_format, features.math_matches, features.matched_domains)
        max_risk_weight = max(PredictiveRouterConfig.RISK_WEIGHTS.values())
        risk_score = min(risk_score_raw / max_risk_weight, 1.0) if max_risk_weight > 0 else 0.0
        expected_failure_modes = self._get_expected_failure_modes(query_lower, response_format, features.math_matches, features.matched_domains)
        
        # 5. Three-Zone model
        if normalized_score < PredictiveRouterConfig.ZONE_LOCAL_LIMIT:
            routing_zone = "Definitely Local"
            decision = "local"
        elif normalized_score >= PredictiveRouterConfig.ZONE_REMOTE_LIMIT:
            routing_zone = "Definitely Remote"
            decision = "remote"
        else:
            routing_zone = "Borderline"
            if risk_score >= PredictiveRouterConfig.RISK_THRESHOLD:
                decision = "remote"
            else:
                decision = "local"
                
        # Confidence logic based on normalized threshold distance
        threshold = (PredictiveRouterConfig.ZONE_LOCAL_LIMIT + PredictiveRouterConfig.ZONE_REMOTE_LIMIT) / 2.0
        max_distance = max(threshold, 1.0 - threshold)
        distance = abs(normalized_score - threshold)
        confidence = 0.5 + 0.5 * (distance / max_distance)
        
        return ComplexityReport(
            score=score,
            normalized_score=normalized_score,
            risk_score=risk_score,
            confidence=confidence,
            decision=decision,
            routing_zone=routing_zone,
            matched_features=matched_features,
            matched_domains=features.matched_domains,
            estimated_output_size=features.output_size,
            estimated_task_count=max(features.task_count, 1),
            comparison_dimension_count=features.comp_dimension_count,
            reasoning_depth=features.reasoning_depth,
            expected_failure_modes=expected_failure_modes,
            matched_entities=features.matched_entities,
            comp_entities_count=features.comp_entities_count,
            comparison_count=features.comparison_count
        )

    def _predict_complexity(self, query: str, response_format: str = "text") -> bool:
        """
        Predicts query complexity using fast, rule-based heuristics to avoid
        local LLM classification overhead (saving CPU/GPU resources).
        """
        # If format is python, route to remote immediately as small local models are not reliable for coding
        if response_format.lower() == "python":
            return True

        report = self.analyze_complexity(query, response_format)
        
        print(f"\n   [Predictive Router] Complexity & Risk Analysis:")
        print(f"      - Query Score: {report.score:.2f} | Normalized: {report.normalized_score:.2f} | Risk Score: {report.risk_score:.2f}")
        print(f"      - Zone: {report.routing_zone} | Decision: {report.decision} | Confidence: {report.confidence:.2f}")
        print(f"      - Subtasks: {report.estimated_task_count} | Output Size: {report.estimated_output_size}")
        print(f"      - Matched Domains: {report.matched_domains}")
        print(f"      - Matched Features: {report.matched_features}")
        if report.expected_failure_modes:
            print(f"      - Predicted Local Failure Modes: {report.expected_failure_modes}")
        print()

        return report.decision == "remote"
