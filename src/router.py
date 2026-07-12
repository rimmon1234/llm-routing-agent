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

_LOCAL_SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n\n"
    "Follow the user's instructions exactly.\n\n"
    "Answer directly without introductions, conclusions, or conversational filler.\n\n"
    "Match the requested format, level of detail, and constraints.\n\n"
    "When the prompt asks for a brief answer, keep it brief.\n"
    "When it asks for explanation, provide only the necessary explanation.\n\n"
    "Verify calculations before responding. If arithmetic is involved, recompute the final answer before producing it.\n\n"
    "Do not add markdown headings, summaries, or extra information unless requested."
)

@dataclass
class RoutingDiagnostics:
    query: str
    total_complexity_score: float
    feature_scores: dict[str, float]
    selected_route: str
    threshold_used: float
    explanation: str
    feature_contributions_version: int = 1
    feature_contributions: dict[str, float] = field(default_factory=dict)
    risk_components: dict[str, float] = field(default_factory=dict)
    risk_weights: dict[str, float] = field(default_factory=dict)

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
    diagnostics: RoutingDiagnostics = None

    def __iter__(self):
        yield self
        yield self.diagnostics

    def __getitem__(self, index):
        if index == 0:
            return self
        elif index == 1:
            return self.diagnostics
        raise IndexError("ComplexityReport has only 2 items when indexed as a tuple")

    def __len__(self):
        return 2

@dataclass
class RoutingFeatures:
    task_count: int
    constraint_count: int
    entities_count: int
    tech_depth: float
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

    @classmethod
    def load_calibrated_thresholds(cls):
        import os
        import json
        path = "calibrated_thresholds.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                def get_threshold(key):
                    # Check top-level
                    val = data.get(key)
                    if val is not None:
                        return val
                    # Check under metadata.calibrated_thresholds
                    metadata = data.get("metadata")
                    if isinstance(metadata, dict):
                        calib = metadata.get("calibrated_thresholds")
                        if isinstance(calib, dict) and calib.get(key) is not None:
                            return calib.get(key)
                    # Check under top-level calibrated_thresholds
                    calib = data.get("calibrated_thresholds")
                    if isinstance(calib, dict) and calib.get(key) is not None:
                        return calib.get(key)
                    return None

                local_limit = get_threshold("ZONE_LOCAL_LIMIT")
                if isinstance(local_limit, (int, float)) and 0.0 <= local_limit <= 1.0:
                    cls.ZONE_LOCAL_LIMIT = float(local_limit)
                    
                remote_limit = get_threshold("ZONE_REMOTE_LIMIT")
                if isinstance(remote_limit, (int, float)) and 0.0 <= remote_limit <= 1.0:
                    cls.ZONE_REMOTE_LIMIT = float(remote_limit)
                    
                risk_threshold = get_threshold("RISK_THRESHOLD")
                if isinstance(risk_threshold, (int, float)) and 0.0 <= risk_threshold <= 1.0:
                    cls.RISK_THRESHOLD = float(risk_threshold)
            except Exception:
                # Fallback to default values on parse error
                pass

    RISK_COMPONENTS_WEIGHTS = {}
    RISK_NORMALIZATION = {}
    EVALUATOR_HISTORICAL_SUCCESS = {}

    @classmethod
    def load_routing_weights(cls):
        import os
        import json
        
        # Default fallback values
        cls.RISK_COMPONENTS_WEIGHTS = {
            "tech_depth": 3.0,
            "reasoning_depth": 2.5,
            "ambiguity": 2.0,
            "constraint_density": 2.0,
            "output_size": 1.5,
            "math_reasoning": 2.0,
            "evaluator_uncertainty": 2.5,
            "distributed_systems": 1.5,
            "json_formatting": 1.0
        }
        cls.RISK_NORMALIZATION = {
            "tech_depth": 5.0,
            "reasoning_depth": 5.0,
            "constraint_density": 5.0,
            "ambiguity": 3.0,
            "math_reasoning": 3.0
        }
        cls.EVALUATOR_HISTORICAL_SUCCESS = {
            "CodeGenerationEvaluator": 0.82,
            "DebuggingEvaluator": 0.85,
            "JSONEvaluator": 0.90,
            "PythonEvaluator": 0.92,
            "ComparisonEvaluator": 0.98,
            "SummarizationEvaluator": 0.95,
            "ArchitectureEvaluator": 0.88,
            "GenericTextEvaluator": 0.95
        }
        
        path = "routing_weights.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Update configurations if present
                if "risk_weights" in data and isinstance(data["risk_weights"], dict):
                    for subk, subv in data["risk_weights"].items():
                        if isinstance(subv, (int, float)):
                            cls.RISK_COMPONENTS_WEIGHTS[subk] = float(subv)
                            
                if "risk_normalization" in data and isinstance(data["risk_normalization"], dict):
                    for subk, subv in data["risk_normalization"].items():
                        if isinstance(subv, (int, float)):
                            cls.RISK_NORMALIZATION[subk] = float(subv)
            except Exception:
                pass
                
        # Load from evaluation_stats.json if it exists
        stats_path = "evaluation_stats.json"
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                for name, entry in stats_data.items():
                    if isinstance(entry, dict) and "success_rate" in entry:
                        cls.EVALUATOR_HISTORICAL_SUCCESS[name] = float(entry["success_rate"])
                    elif isinstance(entry, (int, float)):
                        cls.EVALUATOR_HISTORICAL_SUCCESS[name] = float(entry)
            except Exception:
                pass

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

    KEYWORD_WEIGHTS = {
        # Basic programming keywords (contribute 0.2 weight)
        "python": 0.2, "java": 0.2, "c++": 0.2, "cpp": 0.2, "javascript": 0.2, "js": 0.2,
        "typescript": 0.2, "ts": 0.2, "rust": 0.2, "go": 0.2, "golang": 0.2, "c": 0.2,
        "string": 0.2, "function": 0.2, "method": 0.2, "class": 0.2, "code": 0.2,
        "program": 0.2, "loop": 0.2, "array": 0.2, "list": 0.2, "variable": 0.2, "reverse": 0.2,
        "check": 0.2, "sum": 0.2, "compare": 0.2, "count": 0.2, "palindrome": 0.2, "map": 0.2,
        
        # High-complexity programming concepts (contribute 2.0 weight)
        "compiler": 2.0, "concurrency": 2.0, "optimization": 2.0, "distributed": 2.0,
        "distributed systems": 2.0, "consensus": 2.0, "raft": 2.0, "paxos": 2.0, "compilation": 2.0, 
        "ast": 2.0, "recursion": 1.5, "multithreading": 2.0, "garbage collection": 2.0, 
        "memory leak": 2.0, "leak": 2.0, "deadlock": 2.0, "race condition": 2.0, 
        "synchronization": 2.0, "parallelism": 2.0, "sharding": 2.0, "replication": 2.0, 
        "consistency": 2.0, "mutex": 2.0, "virtual memory": 2.0, "memory management": 2.0,
        "advanced algorithms": 2.0, "compiler design": 2.0, "parallel distributed optimization": 3.0,
        "compiler optimization": 2.5
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
        "Programming": {"python", "java", "c++", "cpp", "javascript", "js", "typescript", "ts", "rust", "compilation", "ast", "recursion", "multithreading", "garbage collection", "string", "function", "method", "class", "code", "program", "loop", "array", "list", "variable", "reverse", "compiler", "concurrency", "optimization", "distributed", "distributed systems", "consensus", "raft", "paxos", "memory leak", "leak", "deadlock", "race condition", "synchronization", "parallelism", "mutex", "virtual memory", "memory management", "advanced algorithms", "compiler design", "parallel distributed optimization", "compiler optimization"},
        "Databases": {"sql", "postgres", "mysql", "indexing", "normalization", "nosql", "sharding", "acid", "transaction", "isolation level", "query planner", "vector database"},
        "Distributed Systems": {"consensus", "raft", "paxos", "sharding", "replication", "consistency", "availability", "distributed", "partition", "split brain", "leader election", "consumer groups"},
        "Networking": {"tcp", "udp", "http", "dns", "latency", "bandwidth", "packet", "routing", "cidr", "socket", "tls", "ssl"},
        "Security": {"encryption", "decryption", "cryptography", "ssl", "tls", "auth", "jwt", "xss", "csrf", "vulnerability", "buffer overflow", "access token", "authorization code"},
        "Artificial Intelligence": {"gradient descent", "neural network", "transformer", "backpropagation", "supervised", "unsupervised", "dataset", "llm", "embeddings"},
        "Cloud": {"kubernetes", "k8s", "docker", "container", "microservices", "aws", "azure", "gcp", "serverless"},
        "Operating Systems": {"kernel", "process", "thread", "mutex", "deadlock", "virtual memory", "scheduling", "sys call", "paging"},
        "Mathematics": {"algebra", "calculus", "derivative", "integral", "matrix", "modulo", "equation", "probability", "statistics", "combinatorics", "proof", "prove", "theorem", "geometry", "trigonometry", "arithmetic", "numerical methods"},
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
        if os.getenv("DOCKER_CONTAINER") == "1":
            return
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
        PredictiveRouterConfig.load_calibrated_thresholds()
        PredictiveRouterConfig.load_routing_weights()
        self.client = client or LLMClient()
        self.evaluator = evaluator or ResponseEvaluator(self.client)
        self.cache = RouterCache(cache_file)
        self._precompile_patterns()
        self.debug_timing = os.getenv("ROUTER_DEBUG_TIMING") == "1"

    def _debug_print_timing(self, message: str) -> None:
        if self.debug_timing:
            print(f"   [Timing] {message}", flush=True)

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
        
        t0 = time.perf_counter()
        local_res = self.client.call_local(query, system_prompt=_LOCAL_SYSTEM_PROMPT, json_mode=use_json_mode)
        t1 = time.perf_counter()
        result["generation_latency_sec"] += (t1 - t0)
        self._debug_print_timing(f"Local generation took {t1 - t0:.2f}s")

        t_tok0 = time.perf_counter()
        pt_tokens = self.client.estimate_tokens(query)
        ct_tokens = self.client.estimate_tokens(local_res)
        t_tok1 = time.perf_counter()
        self._debug_print_timing(f"Token estimation took {t_tok1 - t_tok0:.3f}s")

        result["prompt_tokens_local"] += pt_tokens
        result["completion_tokens_local"] += ct_tokens

        t_eval0 = time.perf_counter()
        eval_result = self.evaluator.evaluate(query, local_res, response_format, schema)
        t_eval1 = time.perf_counter()
        result["evaluation_latency_sec"] += (t_eval1 - t_eval0)
        self._debug_print_timing(f"Local validation evaluation took {t_eval1 - t_eval0:.2f}s")

        is_valid = eval_result.passed if hasattr(eval_result, 'passed') else eval_result[0]
        error_reason = ""
        if not is_valid:
            if hasattr(eval_result, 'critical_failures') and eval_result.critical_failures:
                error_reason = eval_result.critical_failures[0]
            elif hasattr(eval_result, 'failure_reasons') and eval_result.failure_reasons:
                error_reason = eval_result.failure_reasons[0]
            else:
                error_reason = eval_result[1] if len(eval_result) > 1 else ""

        # Fast path: if evaluation passed, return immediately
        if is_valid:
            return local_res, True, ""

        # Estimate repair confidence
        complexity_score = result.get("_complexity_score", 0.3)
        repair_confidence = self._estimate_repair_confidence(
            query, local_res, error_reason, complexity_score, eval_result
        )
        
        if repair_confidence < 0.5:
            print(f"   [Router] Skipping repair (confidence={repair_confidence:.2f} < 0.50, error: {error_reason})")
            return local_res, False, error_reason

        if max_retries > 0 and "Execution error" not in error_reason:
            print(f"   [Router] Repairing (confidence={repair_confidence:.2f}, error: {error_reason})...")
            schema_hint = f"|{schema}" if schema else ""
            for attempt in range(1, max_retries + 1):
                result["local_attempts"] += 1

                repair_prompt = _REPAIR_TMPL.format(
                    q=query, r=local_res, e=error_reason,
                    fmt=response_format, schema=schema_hint
                )
                
                t_tok_rep0 = time.perf_counter()
                pt_rep = self.client.estimate_tokens(repair_prompt)
                t_tok_rep1 = time.perf_counter()
                self._debug_print_timing(f"Repair attempt {attempt} token estimation took {t_tok_rep1 - t_tok_rep0:.3f}s")
                
                result["prompt_tokens_local"] += pt_rep

                t_rep0 = time.perf_counter()
                local_res = self.client.call_local(
                    prompt=repair_prompt,
                    temperature=0.1,
                    json_mode=use_json_mode
                )
                t_rep1 = time.perf_counter()
                self._debug_print_timing(f"Repair attempt {attempt} generation took {t_rep1 - t_rep0:.2f}s")
                
                t_tok_rep_c0 = time.perf_counter()
                ct_rep = self.client.estimate_tokens(local_res)
                t_tok_rep_c1 = time.perf_counter()
                self._debug_print_timing(f"Repair attempt {attempt} completion token estimation took {t_tok_rep_c1 - t_tok_rep_c0:.3f}s")

                result["completion_tokens_local"] += ct_rep

                t_eval_rep0 = time.perf_counter()
                eval_result = self.evaluator.evaluate(query, local_res, response_format, schema)
                t_eval_rep1 = time.perf_counter()
                self._debug_print_timing(f"Repair attempt {attempt} validation took {t_eval_rep1 - t_eval_rep0:.2f}s")

                is_valid = eval_result.passed if hasattr(eval_result, 'passed') else eval_result[0]
                if is_valid:
                    error_reason = ""
                else:
                    if hasattr(eval_result, 'critical_failures') and eval_result.critical_failures:
                        error_reason = eval_result.critical_failures[0]
                    elif hasattr(eval_result, 'failure_reasons') and eval_result.failure_reasons:
                        error_reason = eval_result.failure_reasons[0]
                    else:
                        error_reason = eval_result[1] if len(eval_result) > 1 else ""
                        
                if is_valid:
                    print(f"   {S.good('[Router]')} Repaired attempt {S.good(str(attempt))}!")
                    break
                else:
                    print(f"   {S.warn('[Router]')} Attempt {attempt} failed ({S.warn(error_reason)}).")

        return local_res, is_valid, error_reason

    def _estimate_repair_confidence(self, query: str, response: str, error_reason: str,
                                     complexity_score: float, eval_result) -> float:
        """
        Estimate the probability that a local repair attempt will succeed.
        Returns a value between 0.0 and 1.0.
        
        Factors:
          - base_confidence: inversely proportional to task complexity
          - evaluator_confidence: derived from eval quality score if available
          - syntax_confidence: reduced for degenerate failures (loops, echoing)
          - completeness_score: reduced for empty or very short responses
        """
        # Base: harder tasks are less likely to be repaired successfully
        base_confidence = max(0.1, 1.0 - complexity_score)
        
        # Evaluator confidence: if quality score is available and high, repair is more likely
        evaluator_confidence = 1.0
        if hasattr(eval_result, 'quality_score') and eval_result.quality_score is not None:
            evaluator_confidence = 0.3 + 0.7 * eval_result.quality_score
        
        # Syntax confidence: degenerate failures are unlikely to self-repair
        syntax_confidence = 1.0
        error_lower = error_reason.lower() if error_reason else ""
        if "repetitive" in error_lower or "loop" in error_lower:
            syntax_confidence = 0.1
        elif "echo" in error_lower or "prompt" in error_lower:
            syntax_confidence = 0.15
        elif "too short" in error_lower or "empty" in error_lower:
            syntax_confidence = 0.2
        
        # Completeness: very short responses are likely degenerate
        completeness_score = 1.0
        if response and len(response.strip()) < 20:
            completeness_score = 0.3
        elif response and len(response.strip()) < 50:
            completeness_score = 0.6
            
        confidence = base_confidence * evaluator_confidence * syntax_confidence * completeness_score
        return max(0.0, min(1.0, confidence))

    def _handle_remote_call(self, query: str, result: dict):
        """Call remote API, update result in place. Returns the response."""
        result["remote_attempts"] += 1
        
        t0 = time.perf_counter()
        res, pt, ct = self.client.call_remote(query)
        t1 = time.perf_counter()
        result["generation_latency_sec"] += (t1 - t0)
        self._debug_print_timing(f"Remote Fireworks API call took {t1 - t0:.2f}s")
        
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

    def route_and_execute(self, query: str, strategy: str = "fallback", response_format: str = "text", schema: any = None, max_retries: int = None, no_cache: bool = False) -> dict:
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

        if max_retries is None:
            max_retries = int(os.getenv("MAX_RETRIES", "0"))

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
                result.setdefault("generation_latency_sec", 0.0)
                result.setdefault("evaluation_latency_sec", 0.0)
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
            "cached": False,
            "generation_latency_sec": 0.0,
            "evaluation_latency_sec": 0.0
        }

        use_json_mode = (response_format.lower() == "json")

        if strategy == "always_local":
            result["route_chosen"] = "local"
            result["local_attempts"] = 1
            t_gen0 = time.perf_counter()
            result["response"] = self.client.call_local(query, system_prompt=_LOCAL_SYSTEM_PROMPT, json_mode=use_json_mode)
            result["generation_latency_sec"] += (time.perf_counter() - t_gen0)
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
            is_complex, pred_report = self._predict_complexity(query, response_format)
            result["_complexity_score"] = pred_report.normalized_score

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
        self._debug_print_timing(f"Total route_and_execute request took {result['latency_sec']:.2f}s")

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

    def _get_tech_depth_count(self, query_lower: str) -> float:
        """Calculate weighted tech depth score from matched keywords and phrases."""
        words = set(re.findall(r'\b\w+\b', query_lower))
        score = 0.0
        matched_keywords = set()
        
        # Check phrase match first to prioritize larger composite terms
        for term, weight in sorted(PredictiveRouterConfig.KEYWORD_WEIGHTS.items(), key=lambda x: len(x[0]), reverse=True):
            if ' ' in term or '-' in term:
                if term in query_lower:
                    score += weight
                    # Mask words in matched phrases so they don't get double counted
                    for part in term.split():
                        matched_keywords.add(part)
            else:
                if term in words and term not in matched_keywords:
                    score += weight
                    matched_keywords.add(term)
                    
        # Fall back to general domain keywords that are not explicitly weighted (default weight 1.0)
        for domain, keyword_set in self._domain_word_sets.items():
            for kw in words.intersection(keyword_set):
                if kw not in matched_keywords:
                    score += 1.0
                    matched_keywords.add(kw)
            phrases = self._domain_phrases_list.get(domain, [])
            for phrase in phrases:
                if phrase in query_lower and phrase not in matched_keywords:
                    score += 1.0
                    for part in phrase.split():
                        matched_keywords.add(part)
                        
        return score

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
            
        return depth

    def _get_math_matches(self, query_lower: str) -> int:
        """Count occurrences of mathematical symbols and terms, avoiding false positives on text punctuation."""
        count = 0
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        for sym in PredictiveRouterConfig.MATH_SYMBOLS:
            if len(sym) == 1:
                escaped_sym = re.escape(sym)
                if sym == '-':
                    # Require spaces around hyphen or at least one digit
                    pattern = rf'(?:\b\d+\s*{escaped_sym}\s*\d+\b)|(?:\b[a-zA-Z]\s+{escaped_sym}\s+[a-zA-Z]\b)|(?:\b\d+\s*{escaped_sym}\s+[a-zA-Z]\b)|(?:\b[a-zA-Z]\s*{escaped_sym}\s*\d+\b)'
                elif sym == '/':
                    # Require spaces or digit context to avoid path/URL match like "json/xml" or "http://"
                    pattern = rf'(?:\b\d+\s*{escaped_sym}\s*\d+\b)|(?:\b[a-zA-Z]\s+{escaped_sym}\s+[a-zA-Z]\b)'
                else:
                    pattern = rf'(?:\b\d+|\b[a-zA-Z])\s*{escaped_sym}\s*(?:\d+\b|[a-zA-Z]\b)'
                if re.search(pattern, query_lower):
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

    def _calculate_risk_score(self, query_lower: str, response_format: str, math_matches: int, matched_domains: list[str], features: RoutingFeatures = None) -> tuple[float, dict[str, float], dict[str, float]]:
        """Assess the probability of local model output failure using continuous weighted scaling."""
        
        # 1. Access configurations
        weights = PredictiveRouterConfig.RISK_COMPONENTS_WEIGHTS
        norm_constants = PredictiveRouterConfig.RISK_NORMALIZATION
        
        # 2. Extract continuously scaled features
        tech_depth = features.tech_depth if features else 0.0
        is_coding = "Programming" in matched_domains or response_format.lower() in ("python", "json")
        if is_coding:
            norm_tech_depth = min(tech_depth / norm_constants.get("tech_depth", 5.0), 1.0)
        else:
            norm_tech_depth = 0.0
            
        reasoning_depth = features.reasoning_depth if features else 0
        norm_reasoning_depth = min(reasoning_depth / norm_constants.get("reasoning_depth", 5.0), 1.0)
        
        ambiguity_count = features.ambiguity_count if features else 0
        norm_ambiguity = min(ambiguity_count / norm_constants.get("ambiguity", 3.0), 1.0)
        
        constraint_count = features.constraint_count if features else 0
        norm_constraint_density = min(constraint_count / norm_constants.get("constraint_density", 5.0), 1.0)
        
        output_size = (features.output_size if features else "medium").lower()
        if output_size == "high":
            norm_output_size = 1.0
        elif output_size == "medium":
            norm_output_size = 0.5
        else:
            norm_output_size = 0.1
            
        norm_math = min(math_matches / norm_constants.get("math_reasoning", 3.0), 1.0)
        if "Mathematics" in matched_domains:
            norm_math = max(norm_math, 0.5)
            
        # 3. Dynamic Evaluator Confidence (predicted from historical stats)
        from .evaluator import (
            ComparisonEvaluator, SummarizationEvaluator, ArchitectureEvaluator,
            DebuggingEvaluator, JSONEvaluator, PythonEvaluator, CodeGenerationEvaluator
        )
        
        applicable_evals = []
        if ComparisonEvaluator.is_applicable(query_lower):
            applicable_evals.append("ComparisonEvaluator")
        if SummarizationEvaluator.is_applicable(query_lower):
            applicable_evals.append("SummarizationEvaluator")
        if ArchitectureEvaluator.is_applicable(query_lower):
            applicable_evals.append("ArchitectureEvaluator")
        if DebuggingEvaluator.is_applicable(query_lower):
            applicable_evals.append("DebuggingEvaluator")
        if JSONEvaluator.is_applicable(response_format):
            applicable_evals.append("JSONEvaluator")
        if PythonEvaluator.is_applicable(response_format):
            applicable_evals.append("PythonEvaluator")
        if CodeGenerationEvaluator.is_applicable(query_lower, response_format):
            applicable_evals.append("CodeGenerationEvaluator")
            
        if not applicable_evals:
            applicable_evals.append("GenericTextEvaluator")
            
        success_rates = [
            PredictiveRouterConfig.EVALUATOR_HISTORICAL_SUCCESS.get(name, 0.95)
            for name in applicable_evals
        ]
        predicted_eval_confidence = sum(success_rates) / len(success_rates) if success_rates else 0.95
        norm_evaluator_uncertainty = 1.0 - predicted_eval_confidence
        
        # 4. JSON structure and distributed systems indicators
        norm_json = 1.0 if (response_format.lower() == "json" or "json" in query_lower) else 0.0
        is_distributed = "Distributed Systems" in matched_domains or any(keyword in query_lower for keyword in ["raft", "paxos", "consensus", "distributed", "concurrency", "multithreading", "thread-safe", "mutex", "race condition"])
        norm_distributed = 1.0 if is_distributed else 0.0
        
        # 5. Weighted combination of risk contributions
        contributions = {
            "tech_depth": norm_tech_depth * weights.get("tech_depth", 3.0),
            "reasoning_depth": norm_reasoning_depth * weights.get("reasoning_depth", 2.5),
            "ambiguity": norm_ambiguity * weights.get("ambiguity", 2.0),
            "constraint_density": norm_constraint_density * weights.get("constraint_density", 2.0),
            "output_size": norm_output_size * weights.get("output_size", 1.5),
            "math_reasoning": norm_math * weights.get("math_reasoning", 2.0),
            "evaluator_uncertainty": norm_evaluator_uncertainty * weights.get("evaluator_uncertainty", 2.5),
            "distributed_systems": norm_distributed * weights.get("distributed_systems", 1.5),
            "json_formatting": norm_json * weights.get("json_formatting", 1.0)
        }
        
        weighted_sum = sum(contributions.values())
        total_weight = sum(weights.values())
        
        normalized_risk = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate raw feature scores showing each component's contribution
        risk_components = {k: round(v / total_weight, 4) if total_weight > 0 else 0.0 for k, v in contributions.items()}
        
        return normalized_risk, risk_components, dict(weights)

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

    def _evaluate_complexity_python(self, query_lower: str, response_format: str) -> tuple[float, list[str]]:
        """Evaluate direct complexity score increase from Python target language coding query."""
        score = 0.0
        features = []
        if response_format.lower() == "python" or "python" in query_lower:
            score += PredictiveRouterConfig.COMPLEXITY_WEIGHTS["tech_depth"]
            features.append("complexity_python")
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
        
        # Initialize default feature scores for all features to 0.0
        feature_scores = {
            f"task_count_crossover({features.task_count})": 0.0,
            f"constraints({features.constraint_count})": 0.0,
            f"multi_entity({features.entities_count})": 0.0,
            f"tech_depth({features.tech_depth})": 0.0,
            "output_size_high": 0.0,
            "output_size_medium": 0.0,
            f"ambiguity({features.ambiguity_count})": 0.0,
            f"reasoning_depth({features.reasoning_depth})": 0.0,
            f"math_references({features.math_matches})": 0.0,
            f"domain_breadth({len(features.matched_domains)})": 0.0,
            "length_long": 0.0,
            "length_medium": 0.0,
            "lexical_diversity_high": 0.0,
            "complexity_math": 0.0,
            "complexity_non_python": 0.0,
            "complexity_python": 0.0,
            "complexity_debugging": 0.0,
            "complexity_distributed_systems": 0.0,
            "complexity_large_document": 0.0
        }
        
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
            self._evaluate_complexity_python(query_lower, response_format),
            self._evaluate_complexity_debugging(query_lower, features.matched_domains),
            self._evaluate_complexity_distributed_systems(query_lower, features.matched_domains),
            self._evaluate_complexity_large_document(query_lower)
        ]
        
        for feat_score, feat_names in evaluators:
            score += feat_score
            matched_features.extend(feat_names)
            for name in feat_names:
                feature_scores[name] = feat_score
            
        # 3. Dynamic Normalization boundary
        max_possible = sum(PredictiveRouterConfig.COMPLEXITY_WEIGHTS.values())
        normalized_score = min(score / max_possible, 1.0)
        # 4. Risk estimation
        risk_score, risk_components, risk_weights = self._calculate_risk_score(
            query_lower, 
            response_format, 
            features.math_matches, 
            features.matched_domains, 
            features=features
        )
        expected_failure_modes = self._get_expected_failure_modes(query_lower, response_format, features.math_matches, features.matched_domains)
        
        # 5. Three-Zone model
        threshold_used = 0.0
        explanation_parts = []
        if normalized_score < PredictiveRouterConfig.ZONE_LOCAL_LIMIT:
            routing_zone = "Definitely Local"
            decision = "local"
            threshold_used = PredictiveRouterConfig.ZONE_LOCAL_LIMIT
            explanation_parts.append(f"Normalized score {normalized_score:.2f} is below Definitely Local limit ({PredictiveRouterConfig.ZONE_LOCAL_LIMIT:.2f}).")
        elif normalized_score >= PredictiveRouterConfig.ZONE_REMOTE_LIMIT:
            routing_zone = "Definitely Remote"
            decision = "remote"
            threshold_used = PredictiveRouterConfig.ZONE_REMOTE_LIMIT
            explanation_parts.append(f"Normalized score {normalized_score:.2f} is at or above Definitely Remote limit ({PredictiveRouterConfig.ZONE_REMOTE_LIMIT:.2f}).")
        else:
            routing_zone = "Borderline"
            explanation_parts.append(f"Normalized score {normalized_score:.2f} is in Borderline zone [{PredictiveRouterConfig.ZONE_LOCAL_LIMIT:.2f}, {PredictiveRouterConfig.ZONE_REMOTE_LIMIT:.2f}).")
            if risk_score >= PredictiveRouterConfig.RISK_THRESHOLD:
                decision = "remote"
                threshold_used = PredictiveRouterConfig.RISK_THRESHOLD
                explanation_parts.append(f"Risk score {risk_score:.2f} is at or above threshold ({PredictiveRouterConfig.RISK_THRESHOLD:.2f}). Routing to remote.")
            else:
                decision = "local"
                threshold_used = PredictiveRouterConfig.RISK_THRESHOLD
                explanation_parts.append(f"Risk score {risk_score:.2f} is below threshold ({PredictiveRouterConfig.RISK_THRESHOLD:.2f}). Routing to local.")
                
        # Confidence logic based on normalized threshold distance
        threshold = (PredictiveRouterConfig.ZONE_LOCAL_LIMIT + PredictiveRouterConfig.ZONE_REMOTE_LIMIT) / 2.0
        max_distance = max(threshold, 1.0 - threshold)
        distance = abs(normalized_score - threshold)
        confidence = 0.5 + 0.5 * (distance / max_distance)
        
        explanation = " ".join(explanation_parts)
        
        diagnostics = RoutingDiagnostics(
            query=query,
            total_complexity_score=score,
            feature_scores=feature_scores,
            selected_route=decision,
            threshold_used=threshold_used,
            explanation=explanation,
            feature_contributions_version=1,
            feature_contributions={
                "tech_depth": float(features.tech_depth),
                "math_score": float(features.math_matches),
                "reasoning_depth": float(features.reasoning_depth),
                "constraint_score": float(features.constraint_count),
                "task_count": float(features.task_count),
                "ambiguity": float(features.ambiguity_count),
                "entity_count": float(features.entities_count),
                "complexity": float(normalized_score),
                "risk": float(risk_score),
            },
            risk_components=risk_components,
            risk_weights=risk_weights
        )
        
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
            comparison_count=features.comparison_count,
            diagnostics=diagnostics
        )

    def _predict_complexity(self, query: str, response_format: str = "text") -> tuple:
        """
        Predicts query complexity using fast, rule-based heuristics to avoid
        local LLM classification overhead (saving CPU/GPU resources).
        Returns (is_complex: bool, report: ComplexityReport).
        """
        report = self.analyze_complexity(query, response_format)
        
        print(f"\n   [Predictive Router] Complexity & Risk Analysis:")
        print(f"      - Query Score: {report.score:.2f} | Normalized: {report.normalized_score:.2f} | Risk Score: {report.risk_score:.2f}")
        print(f"      - Zone: {report.routing_zone} | Decision: {report.decision} | Confidence: {report.confidence:.2f}")
        print(f"      - Subtasks: {report.estimated_task_count} | Output Size: {report.estimated_output_size}")
        print(f"      - Matched Domains: {report.matched_domains}")
        print(f"      - Matched Features: {report.matched_features}")
        if report.diagnostics and report.diagnostics.risk_components:
            print(f"      - Risk Components: {report.diagnostics.risk_components}")
            print(f"      - Risk Weights: {report.diagnostics.risk_weights}")
        if report.expected_failure_modes:
            print(f"      - Predicted Local Failure Modes: {report.expected_failure_modes}")
        print()

        return report.decision == "remote", report
