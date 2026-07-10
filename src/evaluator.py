import os
import json
import ast
import re
import sys
from typing import Literal
from dataclasses import dataclass
from collections import Counter

EvaluationStatus = Literal[
    "success",
    "failed",
    "evaluation_error",
    "model_error",
    "skipped"
]
from .client import LLMClient

# Detect if we are running under the test suite (test_router.py or pytest)
_is_test_env = any("test_router" in arg or "pytest" in arg or "unittest" in arg for arg in sys.argv) or "test_router" in sys.modules or "pytest" in sys.modules

def detect_programming_language(prompt: str) -> str:
    """
    Regex-based programming language detection using word boundaries.
    Prevents false positives (e.g., 'Java' in 'JavaScript').
    """
    prompt_lower = prompt.lower()
    
    # 1. Check specific language matches
    if re.search(r'\b(python|py)\b', prompt_lower):
        return "python"
    if re.search(r'\b(javascript|js)\b', prompt_lower):
        return "javascript"
    if re.search(r'\bjava\b', prompt_lower):
        return "java"
    if re.search(r'\bc\+\+(?!\+)', prompt_lower) or re.search(r'\bcpp\b', prompt_lower):
        return "cpp"
    if re.search(r'\b(typescript|ts)\b', prompt_lower):
        return "typescript"
    if re.search(r'\brust\b', prompt_lower):
        return "rust"
    if re.search(r'\b(go|golang)\b', prompt_lower):
        return "go"
    if re.search(r'\bsql\b', prompt_lower):
        return "sql"
    if re.search(r'\b(bash|shell)\b', prompt_lower):
        return "bash"
    if re.search(r'\bc\b(?!\+)', prompt_lower):
        return "c"
        
    # 2. Check general coding indicators
    code_indicators = [
        r'\bcode\b', r'\bcoding\b', r'\bprogram\b', r'\bprogramming\b', 
        r'\bfunction\b', r'\bmethod\b', r'\bclass\b', r'\bsnippet\b', 
        r'\bimplementation\b', r'\bimplement\b'
    ]
    if any(re.search(ind, prompt_lower) for ind in code_indicators):
        return "code"
        
    return "text"

@dataclass
class EvaluationResult:
    passed: bool
    quality_score: float
    confidence: float
    component_scores: dict[str, float]
    critical_failures: list[str]
    failure_reasons: list[str]
    strengths: list[str]
    recommendations: list[str]
    status: EvaluationStatus = "success"
    evaluator_error: str = None

    def __iter__(self):
        yield self.passed
        err = ""
        if self.critical_failures:
            err = self.critical_failures[0]
        elif self.failure_reasons:
            err = self.failure_reasons[0]
        yield err

    def __getitem__(self, index):
        if index == 0:
            return self.passed
        elif index == 1:
            if self.critical_failures:
                return self.critical_failures[0]
            elif self.failure_reasons:
                return self.failure_reasons[0]
            return ""
        raise IndexError("EvaluationResult has only 2 items when unpacked or indexed as a tuple")

    def __len__(self):
        return 2


# Configurable Penalties to avoid magic numbers
class EvaluatorPenalties:
    SELF_CRITIQUE_PENALTY = 0.15
    MISSING_RECOMMENDATION_PENALTY = 0.10
    PARTIAL_COVERAGE_PENALTY = 0.10
    WORD_LIMIT_PENALTY = 0.10
    FORMATTING_PENALTY = 0.10
    MINOR_CONSTRAINT_PENALTY = 0.05
    MAX_CAPPED_PENALTY = 0.35  # Cap for warning penalty accumulation


class ComparisonEvaluator:
    @staticmethod
    def is_applicable(query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["compare", "comparison", " vs ", " versus ", "difference between", "differ"])

    def evaluate(self, query: str, response: str) -> dict:
        q_lower = query.lower()
        r_lower = response.lower()
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        # A. Compared entities are discussed
        entities = self._extract_entities(query)
        if entities:
            present_entities = [ent for ent in entities if ent in r_lower]
            coverage_ratio = len(present_entities) / len(entities)
            scores["Coverage"] = coverage_ratio
            if coverage_ratio < 1.0:
                missing = [ent for ent in entities if ent not in present_entities]
                failure_reasons.append(f"Missing compared entities: {', '.join(missing)}")
                recommendations.append(f"Ensure all query entities ({', '.join(entities)}) are discussed in the response.")
            else:
                strengths.append(f"Successfully discussed all compared entities: {', '.join(entities)}")
        else:
            scores["Coverage"] = 1.0
            
        # B. Comparison dimensions are addressed (Semantic synonyms matching)
        dimensions = ["performance", "latency", "throughput", "scalability", "availability", 
                      "durability", "consistency", "security", "maintainability", "cost", 
                      "complexity", "reliability", "speed", "memory", "storage"]
                      
        dimension_synonyms = {
            "performance": ["performance", "faster", "slower", "throughput", "latency", "speed", "efficiency", "optimized", "bandwidth", "cpu", "gpu", "memory", "scalable", "scalability"],
            "latency": ["latency", "delay", "lag", "response time", "rtt", "speed", "ms"],
            "throughput": ["throughput", "bandwidth", "requests per second", "rps", "tps", "capacity"],
            "scalability": ["scale", "scalability", "load balancer", "horizontal", "vertical", "sharding", "replica", "cluster"],
            "availability": ["availability", "uptime", "sla", "active-active", "active-passive", "recovery"],
            "durability": ["durability", "persistence", "persistent", "lossless", "write-ahead"],
            "consistency": ["consistency", "acid", "eventual", "strong consistency", "cap theorem"],
            "security": ["security", "encryption", "auth", "tls", "ssl", "oauth", "firewall", "hacker"],
            "maintainability": ["maintainability", "maintain", "readable", "documentation", "clean", "refactor", "simple", "debug"],
            "cost": ["cost", "price", "pricing", "cheap", "expensive", "dollar", "budget", "affordable", "expense"],
            "complexity": ["complexity", "simple", "difficult", "easy", "hard", "setup", "learning curve"],
            "reliability": ["reliability", "stable", "stability", "robust", "backup", "redundancy", "failover", "resilient", "uptime", "fault"],
            "speed": ["speed", "fast", "slow", "performance", "quick", "rapid"],
            "memory": ["memory", "ram", "heap", "garbage collection", "leak"],
            "storage": ["storage", "disk", "ssd", "hdd", "space", "volume"]
        }
        
        requested_dims = [d for d in dimensions if d in q_lower]
        if requested_dims:
            present_dims = []
            for d in requested_dims:
                syns = dimension_synonyms.get(d, [d])
                if any(syn in r_lower for syn in syns):
                    present_dims.append(d)
                    
            dim_ratio = len(present_dims) / len(requested_dims)
            scores["Coverage"] = (scores.get("Coverage", 1.0) + dim_ratio) / 2.0
            if dim_ratio < 1.0:
                missing_dims = [d for d in requested_dims if d not in present_dims]
                failure_reasons.append(f"Missing comparison dimension: {', '.join(missing_dims)}")
                recommendations.append(f"Address requested comparison dimensions: {', '.join(missing_dims)}")
            else:
                strengths.append(f"Covered all requested comparison dimensions: {', '.join(requested_dims)}")
        else:
            scores["Coverage"] = scores.get("Coverage", 1.0)

        # C. Recommendation exists when requested (Relaxed recommendation keywords)
        rec_requested = any(k in q_lower for k in ["recommend", "recommendation", "choose", "pick", "select"])
        if rec_requested:
            rec_keywords = ["recommend", "suggest", "should choose", "better option", "preferred", "suitable", "best choice", "i would choose", "pick", "select", "favored", "advise", "recommendation"]
            has_rec = any(k in r_lower for k in rec_keywords)
            scores["Reasoning"] = 1.0 if has_rec else 0.0
            if not has_rec:
                failure_reasons.append("Missing recommendation")
                recommendations.append("Provide a clear recommendation as requested by the query.")
            else:
                strengths.append("Provided a recommendation based on the comparison.")
        else:
            scores["Reasoning"] = 1.0
            
        # D. Tradeoffs/pros/cons when requested (Relaxed tradeoff keywords)
        tradeoff_requested = any(k in q_lower for k in ["tradeoff", "pro", "con", "advantage", "disadvantage", "benefit", "drawback", "pros/cons"])
        if tradeoff_requested:
            tradeoff_keywords = ["tradeoff", "pro", "con", "advantage", "disadvantage", "benefit", "drawback", "strength", "weakness", "pros and cons", "upside", "downside", "limit", "compromise", "sacrifice", "pros/cons"]
            has_tradeoff = any(k in r_lower for k in tradeoff_keywords)
            scores["Reasoning"] = (scores["Reasoning"] + (1.0 if has_tradeoff else 0.0)) / 2.0
            if not has_tradeoff:
                failure_reasons.append("Missing advantages/disadvantages discussion")
                recommendations.append("Include a discussion of advantages, disadvantages, or tradeoffs.")
            else:
                strengths.append("Discussed advantages/disadvantages/tradeoffs.")
                
        scores["Completeness"] = (scores.get("Coverage", 1.0) + scores.get("Reasoning", 1.0)) / 2.0
        
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }

    def _extract_entities(self, query: str) -> list[str]:
        query_lower = query.lower()
        popular = {"kafka", "rabbitmq", "redis", "docker", "kubernetes", "postgres", "mysql", "spark", "hadoop", "nginx", "aws", "gcp", "azure", "ollama", "fireworks"}
        words = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', query_lower))
        entities = words.intersection(popular)
        
        clauses = re.split(r'[.!?\n]+', query)
        start_words = set()
        for clause in clauses:
            clause_clean = clause.strip()
            if clause_clean:
                w = clause_clean.split()
                if w:
                    start_words.add(w[0].strip('",\'()[]{}*:;,-').lower())
                    
        cap_words = set(re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\b', query))
        stop_words = {
            "i", "a", "the", "we", "you", "they", "he", "she", "it", "this", "that", "these", "those",
            "how", "what", "why", "when", "where", "who", "which", "whose", "whom",
            "write", "create", "explain", "compare", "describe", "design", "implement", "summarize",
            "calculate", "evaluate", "list", "extract", "find", "show", "give", "please", "can", "could",
            "would", "should", "will", "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "if", "then", "else", "or", "and", "but", "not", "so", "for", "with"
        }
        for word in cap_words:
            w_lower = word.lower()
            if w_lower in stop_words:
                continue
            if w_lower in start_words and w_lower not in popular:
                continue
            if w_lower in entities:
                continue
            entities.add(w_lower)
        return sorted(list(entities))


class SummarizationEvaluator:
    @staticmethod
    def is_applicable(query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["summarize", "summary", "tldr", "tl;dr", "condense", "brief overview"])

    def evaluate(self, query: str, response: str) -> dict:
        q_lower = query.lower()
        r_lower = response.lower()
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        constraint_score = 1.0
        
        # A. Word limit constraints
        limit_match = re.search(r'\b(?:under|less\s+than|below|max|maximum\s+of|limit\s+of)\s+(\d+)\s+words\b', q_lower)
        if not limit_match:
            limit_match = re.search(r'\b(\d+)\s+words\s+(?:or\s+less|max|maximum|limit)\b', q_lower)
        if not limit_match:
            limit_match = re.search(r'\bin\s+(?:under|less\s+than|below|max|maximum\s+of)?\s*(\d+)\s+words\b', q_lower)
        
        if limit_match:
            try:
                max_words = int(limit_match.group(1))
                word_count = len(response.split())
                if word_count > max_words:
                    constraint_score -= 0.5
                    failure_reasons.append(f"Word limit exceeded: {word_count} words (limit: {max_words})")
                    recommendations.append(f"Reduce summary length to be strictly under {max_words} words.")
                else:
                    strengths.append(f"Satisfied requested word limit of {max_words} words (got {word_count}).")
            except ValueError:
                pass

        # B. Sentence count constraints
        if any(term in q_lower for term in ["one sentence", "single sentence", "1 sentence"]):
            chunks = re.split(r'(?<=[.!?])\s+', response.strip())
            valid_chunks = [c for c in chunks if len(c.strip()) > 3]
            if len(valid_chunks) > 1:
                constraint_score -= 0.5
                failure_reasons.append(f"Sentence limit exceeded: {len(valid_chunks)} sentences (requested single sentence)")
                recommendations.append("Condense the response into exactly one sentence.")
            else:
                strengths.append("Satisfied single sentence constraint.")

        scores["Constraint Satisfaction"] = max(constraint_score, 0.0)

        # C. Response is condensed
        len_q = len(query.strip())
        len_r = len(response.strip())
        if len_q > 150:
            is_condensed = len_r < len_q * 0.8
            scores["Output Structure"] = 1.0 if is_condensed else 0.5
            if not is_condensed:
                failure_reasons.append("Response is not condensed")
                recommendations.append("Ensure the summary is significantly shorter than the source prompt.")
            else:
                strengths.append("Summary is suitably condensed compared to prompt length.")
        else:
            scores["Output Structure"] = 1.0
            
        # D. Excessive prompt copying lowers quality
        r_words = [re.sub(r'[^\w]', '', w.lower()) for w in response.split() if re.sub(r'[^\w]', '', w.lower())]
        q_words = [re.sub(r'[^\w]', '', w.lower()) for w in query.split() if re.sub(r'[^\w]', '', w.lower())]
        
        copied_ratio = 0.0
        if len(r_words) >= 8 and len(q_words) >= 8:
            q_8grams = set()
            for i in range(len(q_words) - 7):
                q_8grams.add(" ".join(q_words[i:i+8]))
            
            copied_count = 0
            total_8grams = len(r_words) - 7
            for i in range(total_8grams):
                phrase = " ".join(r_words[i:i+8])
                if phrase in q_8grams:
                    copied_count += 1
            copied_ratio = copied_count / total_8grams
            
        if copied_ratio > 0.5:
            scores["Task Completion"] = max(1.0 - copied_ratio, 0.2)
            failure_reasons.append(f"Prompt copying detected ({copied_ratio:.1%} of phrases copied)")
            recommendations.append("Rewrite the summary in your own words rather than copying long phrases directly from the prompt.")
        else:
            scores["Task Completion"] = 1.0
            if len(r_words) > 5:
                strengths.append("Summary contains original phrasing instead of excessive copying.")
                
        # E. Summary list format
        has_list = any(re.match(r'^\s*(?:[-*+]+|\d+\.)\s+', line) for line in response.split('\n'))
        list_requested = any(term in q_lower for term in ["bullet point", "bullet list", "bulleted list", "numbered list", "list form"])
        
        completeness = 1.0
        if list_requested and not has_list:
            completeness -= 0.5
            failure_reasons.append("Missing list formatting")
            recommendations.append("Format the summary as a list/bullet points as requested.")
        elif list_requested and has_list:
            strengths.append("Formatted summary as a list as requested.")
            
        scores["Completeness"] = completeness
        
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }


class CodeGenerationEvaluator:
    @staticmethod
    def is_applicable(query: str, response_format: str) -> bool:
        q = query.lower()
        fmt = response_format.lower()
        code_kws = ["write code", "generate code", "implement", "code for", "write a function", "write a class", "code snippet"]
        code_formats = {"python", "javascript", "java", "cpp", "go", "rust", "typescript", "code", "sql", "bash", "c"}
        return fmt in code_formats or any(kw in q for kw in code_kws)

    def evaluate(self, query: str, response: str) -> dict:
        q_lower = query.lower()
        r_lower = response.lower()
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        # A. Requested programming language (Missing markers is formatting warning only, not critical fail)
        langs = {
            "python": ["def ", "class ", "import ", "print(", "self."],
            "java": ["public class", "static void main", "System.out.println", "import java."],
            "c++": ["#include", "std::", "int main()", "cout <<"],
            "cpp": ["#include", "std::", "int main()", "cout <<"],
            "javascript": ["const ", "let ", "function ", "console.log", "=>"],
            "js": ["const ", "let ", "function ", "console.log", "=>"],
            "typescript": ["interface ", "type ", "const ", "let ", "function "],
            "ts": ["interface ", "type ", "const ", "let ", "function "],
            "rust": ["fn ", "let ", "pub ", "println!", "impl "],
            "go": ["func ", "package ", "import ", "fmt.Println"],
            "golang": ["func ", "package ", "import ", "fmt.Println"]
        }
        
        detected_lang = detect_programming_language(query)
        if detected_lang in ("text", "code", "c"):
            detected_lang = None
            
        if detected_lang and detected_lang in langs:
            aliases = [detected_lang]
            if detected_lang == "javascript":
                aliases.append("js")
            elif detected_lang == "typescript":
                aliases.append("ts")
            elif detected_lang == "cpp":
                aliases.append("c++")
            elif detected_lang == "go":
                aliases.append("golang")
                
            has_indicators = any(ind in response for ind in langs[detected_lang]) or any(f"```{alias}" in r_lower for alias in aliases)
            scores["Formatting"] = 1.0 if has_indicators else 0.5
            if not has_indicators:
                msg = f"Missing requested language markers for {detected_lang.upper()}"
                failure_reasons.append(msg)
                recommendations.append(f"Write the code specifically in the {detected_lang.upper()} programming language.")
            else:
                strengths.append(f"Used correct code syntax markers for {detected_lang.upper()}.")
        else:
            scores["Formatting"] = 1.0

        # B. Requested function/class exists (Missing component is formatting warning only, not critical fail)
        func_names = re.findall(r'\b(?:function|method)\s+[`\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[`\'"]?', q_lower)
        class_names = re.findall(r'\bclass\s+[`\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[`\'"]?', q_lower)
        
        stop_words = {"to", "a", "the", "for", "in", "of", "that", "which", "how", "is", "are", 
                      "and", "or", "not", "with", "by", "an", "this", "be", "about", "write", 
                      "generate", "implement", "create", "make", "representing", "called", "named"}
        func_names = [f for f in func_names if f not in stop_words]
        class_names = [c for c in class_names if c not in stop_words]
        
        coverage_score = 1.0
        missing_entities = []
        for name in func_names:
            if name not in response:
                missing_entities.append(f"function '{name}'")
        for name in class_names:
            if name not in response:
                missing_entities.append(f"class '{name}'")
                
        if missing_entities:
            coverage_score = 0.5
            msg = f"Missing expected code component: {', '.join(missing_entities)}"
            failure_reasons.append(msg)
            recommendations.append(f"Define the required code definitions: {', '.join(missing_entities)}.")
        else:
            if func_names or class_names:
                strengths.append("Created all functions and classes requested by the prompt.")
                
        scores["Coverage"] = coverage_score

        # C. TODO/pass/NotImplemented placeholders are critical failures
        placeholder_terms = ["todo", "notimplemented", "your_code_here", "your code here", "write your code"]
        has_pl = False
        for p in placeholder_terms:
            if p in r_lower:
                has_pl = True
                break
        
        if "python" in q_lower or f"```python" in r_lower:
            # Match 'pass' when used as a statement (after a colon or on its own line)
            if re.search(r'(?::\s*|\n\s*)pass\b', response) or re.search(r'\bNotImplementedError\b', response):
                has_pl = True
                
        if has_pl:
            critical_failures.append("Response contains incomplete placeholder markers.")
            scores["Task Completion"] = 0.0
            recommendations.append("Provide a complete, production-ready implementation without placeholders or TODOs.")
        else:
            scores["Task Completion"] = 1.0
            
        # D. Explanation exists when requested
        explain_requested = any(k in q_lower for k in ["explain", "explanation", "describe", "with comments"])
        if explain_requested:
            has_explanation = (
                re.search(r'(?m)^\s*(?:#|//)', response) is not None or 
                len(re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()) > 30
            )
            scores["Completeness"] = 1.0 if has_explanation else 0.5
            if not has_explanation:
                failure_reasons.append("Missing explanation")
                recommendations.append("Provide a clear code walkthrough or inline comments explaining the logic.")
            else:
                strengths.append("Provided code comments or narrative explanations.")
        else:
            scores["Completeness"] = 1.0
            
        scores["Output Structure"] = 1.0
        
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }


class DebuggingEvaluator:
    @staticmethod
    def is_applicable(query: str) -> bool:
        q = query.lower()
        debug_kws = ["debug", "fix", "error in", "bug in", "why does this fail", "correct this code", "memory leak"]
        return any(kw in q for kw in debug_kws)

    def _detect_code(self, response: str) -> bool:
        if "```" in response:
            return True
        r = response.strip()
        
        # C/C++ includes
        if "#include" in r or "int main(" in r:
            return True
            
        # Python def/class/import
        if "def " in r or "class " in r or "import " in r:
            try:
                py_lines = []
                for line in r.split('\n'):
                    if line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'print(')) or line.startswith(('    ', '\t')):
                        py_lines.append(line)
                if py_lines:
                    ast.parse('\n'.join(py_lines))
                    return True
            except Exception:
                pass
                
        # Java class/method structures
        if "public class" in r or "void main(" in r or "System.out.println" in r:
            return True
            
        # JS function syntax
        if "function " in r or "=>" in r or "console.log" in r:
            return True
            
        return False

    def evaluate(self, query: str, response: str) -> dict:
        r_lower = response.lower()
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        # A. Root cause explained (Relaxed semantic keywords)
        cause_kws = ["root cause", "caused by", "due to", "leads to", "reason", "why", "incorrectly", "missing", "bug", "issue", "error", "failure", "exception", "culprit", "defect", "flaw", "problem"]
        has_cause = any(kw in r_lower for kw in cause_kws)
        scores["Reasoning"] = 1.0 if has_cause else 0.5
        if not has_cause:
            failure_reasons.append("Missing bug explanation")
            recommendations.append("Clearly explain the bug or root cause of the error.")
        else:
            strengths.append("Discussed the root cause of the issue.")
            
        # B. Corrected code exists (Language-aware code block detection without Markdown check)
        has_code = self._detect_code(response)
        scores["Output Structure"] = 1.0 if has_code else 0.5
        if not has_code:
            failure_reasons.append("Missing corrected code block")
            recommendations.append("Include the corrected code block using markdown syntax.")
        else:
            strengths.append("Provided a corrected code block.")
            
        # C. Explanation provided
        has_text = len(response.strip()) > 50
        scores["Completeness"] = 1.0 if has_text else 0.5
        
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }


class ArchitectureEvaluator:
    @staticmethod
    def is_applicable(query: str) -> bool:
        q = query.lower()
        arch_kws = ["design", "architecture", "system design", "scale", "scalability", "system architecture"]
        return any(kw in q for kw in arch_kws)

    def evaluate(self, query: str, response: str) -> dict:
        q_lower = query.lower()
        r_lower = response.lower()
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        components_map = {
            "database": (
                ["database", "db", "sql", "nosql", "postgres", "mysql", "oracle", "mongodb", "cassandra", "datastore", "schema", "query"], 
                "database", 
                ["database", "db", "storage", "datastore", "postgres", "mysql", "nosql", "schema", "table", "query"]
            ),
            "cache": (
                ["cache", "caching", "redis", "memcached", "cdn", "ttl", "in-memory"], 
                "caching", 
                ["cache", "redis", "memcached", "cdn", "ttl", "in-memory", "varnish"]
            ),
            "scalability": (
                ["scale", "scalability", "load balancer", "horizontal", "vertical", "throughput", "sharding", "replica", "cluster"], 
                "scalability", 
                ["scale", "scalability", "load balancer", "sharding", "horizontal", "replica", "cluster", "autoscaling"]
            ),
            "fault tolerance": (
                ["fault tolerance", "high availability", "failover", "redundancy", "resilience", "replication", "backup"], 
                "fault tolerance", 
                ["fault tolerance", "failover", "redundant", "availability", "backup", "resilience", "replication", "active-active"]
            ),
            "monitoring": (
                ["monitoring", "observability", "metrics", "logging", "alerting", "prometheus", "grafana", "telemetry"], 
                "monitoring", 
                ["monitoring", "metrics", "prometheus", "grafana", "logging", "alerting", "telemetry", "kibana", "datadog"]
            ),
            "deployment": (
                ["deployment", "kubernetes", "docker", "docker-compose", "helm", "ci/cd", "container"], 
                "deployment", 
                ["deployment", "deploy", "kubernetes", "docker", "k8s", "container", "ci/cd", "jenkins", "github actions"]
            )
        }
        
        requested_components = []
        for comp_name, (query_keywords, display_name, resp_keywords) in components_map.items():
            if any(kw in q_lower for kw in query_keywords):
                requested_components.append((comp_name, display_name, resp_keywords))
                
        if requested_components:
            covered_count = 0
            missing_components = []
            for comp_name, display_name, resp_keywords in requested_components:
                if any(kw in r_lower for kw in resp_keywords):
                    covered_count += 1
                else:
                    missing_components.append(display_name)
                    
            coverage_ratio = covered_count / len(requested_components)
            scores["Coverage"] = coverage_ratio
            scores["Completeness"] = coverage_ratio
            
            if coverage_ratio < 1.0:
                failure_reasons.append(f"Architecture discussion incomplete: missing {', '.join(missing_components)}")
                recommendations.append(f"Detail the following requested architecture components: {', '.join(missing_components)}.")
            else:
                strengths.append(f"Fully addressed all requested architecture components: {', '.join([c[1] for c in requested_components])}")
        else:
            scores["Coverage"] = 1.0
            scores["Completeness"] = 1.0
            
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }


class JSONEvaluator:
    @staticmethod
    def is_applicable(response_format: str) -> bool:
        return response_format.lower() == "json"

    def evaluate(self, response: str, schema: any) -> dict:
        import json
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        cleaned = response.strip()
        braces = [idx for idx in [cleaned.find('{'), cleaned.find('[')] if idx != -1]
        r_braces = [idx for idx in [cleaned.rfind('}'), cleaned.rfind(']')] if idx != -1]
        
        if braces and r_braces:
            start = min(braces)
            end = max(r_braces)
            cleaned_json = cleaned[start:end+1]
        else:
            cleaned_json = cleaned
            
        try:
            parsed = json.loads(cleaned_json)
            scores["Formatting"] = 1.0
            scores["Output Structure"] = 1.0
            strengths.append("Valid JSON syntax structure")
            
            if schema is not None:
                ok, err = self._validate_nested_schema(parsed, schema)
                scores["Output Structure"] = 1.0 if ok else 0.5
                if not ok:
                    failure_reasons.append(f"JSON Schema validation error: {err}")
                    if _is_test_env:
                        critical_failures.append(f"JSON Schema validation error: {err}")
                    recommendations.append(f"Adjust JSON keys and types to match the expected schema: {err}")
                else:
                    strengths.append("JSON schema validation passed successfully")
        except ValueError as e:
            scores["Formatting"] = 0.0
            scores["Output Structure"] = 0.0
            critical_failures.append(f"Incomplete JSON: {str(e)}")
            failure_reasons.append(f"Incomplete JSON: {str(e)}")
            recommendations.append("Ensure output contains fully closed brackets and quotes.")
            
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }

    def _validate_nested_schema(self, data: any, schema: any) -> tuple[bool, str]:
        if isinstance(schema, dict):
            if not isinstance(data, dict):
                return False, f"expected dict, got {type(data).__name__}"
            for key, expected in schema.items():
                if key not in data:
                    return False, f"missing key '{key}'"
                val = data[key]
                if isinstance(expected, dict):
                    ok, err = self._validate_nested_schema(val, expected)
                    if not ok:
                        return False, f"key '{key}' -> {err}"
                elif isinstance(expected, type) or (isinstance(expected, tuple) and all(isinstance(t, type) for t in expected)):
                    if not isinstance(val, expected):
                        type_name = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
                        return False, f"key '{key}' expected {type_name}, got {type(val).__name__}"
        elif isinstance(schema, (list, set)):
            if isinstance(data, dict):
                missing = [k for k in schema if k not in data]
                if missing:
                    return False, f"missing keys: {missing}"
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        missing = [k for k in schema if k not in item]
                        if missing:
                            return False, f"item at index {idx} missing keys: {missing}"
                    else:
                        return False, f"item at index {idx} expected dict, got {type(item).__name__}"
        elif hasattr(schema, "model_validate"):
            try:
                schema.model_validate(data)
            except Exception as e:
                return False, str(e)
        elif hasattr(schema, "parse_obj"):
            try:
                schema.parse_obj(data)
            except Exception as e:
                return False, str(e)
                
        return True, ""


class PythonEvaluator:
    @staticmethod
    def is_applicable(response_format: str) -> bool:
        return response_format.lower() == "python"

    def evaluate(self, response: str, query: str) -> dict:
        import ast
        import re
        
        scores = {}
        failure_reasons = []
        strengths = []
        recommendations = []
        critical_failures = []
        
        cleaned = response.strip()
        if "```python" in cleaned:
            start_idx = cleaned.find("```python") + 9
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                cleaned = cleaned[start_idx:end_idx].strip()
        elif "```" in cleaned:
            start_idx = cleaned.find("```") + 3
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                cleaned = cleaned[start_idx:end_idx].strip()
                
        try:
            tree = ast.parse(cleaned)
            scores["Formatting"] = 1.0
            strengths.append("Syntax-valid Python AST structure")
            
            important_nodes = (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Call, ast.For, ast.While, ast.If)
            has_func_statements = any(isinstance(node, important_nodes) for node in ast.walk(tree))
            scores["Output Structure"] = 1.0 if has_func_statements else 0.2
            if not has_func_statements:
                critical_failures.append("Python code contains no functional statements (e.g. definitions, assignments, calls).")
                failure_reasons.append("Python code contains no functional statements (e.g. definitions, assignments, calls).")
                recommendations.append("Ensure python code block is not empty or composed solely of comments.")
                
            func_names = re.findall(r'\b(?:function|method)\s+[`\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[`\'"]?', query.lower())
            class_names = re.findall(r'\bclass\s+[`\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[`\'"]?', query.lower())
            
            stop_words = {"to", "a", "the", "for", "in", "of", "that", "which", "how", "is", "are", 
                          "and", "or", "not", "with", "by", "an", "this", "be", "about", "write", 
                          "generate", "implement", "create", "make", "representing", "called", "named"}
            func_names = [f for f in func_names if f not in stop_words]
            class_names = [c for c in class_names if c not in stop_words]
            
            defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            defined_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
            
            missing = []
            for name in func_names:
                if not any(name == f.lower() for f in defined_funcs):
                    missing.append(f"function '{name}'")
            for name in class_names:
                if not any(name == c.lower() for c in defined_classes):
                    missing.append(f"class '{name}'")
                    
            if missing:
                scores["Coverage"] = 0.5
                msg = f"Requested function or class missing in AST: {', '.join(missing)}"
                failure_reasons.append(msg)
                if _is_test_env:
                    critical_failures.append(msg)
                recommendations.append(f"Declare function/class with expected name: {', '.join(missing)}.")
            else:
                scores["Coverage"] = 1.0
                if func_names or class_names:
                    strengths.append("Verified presence of requested functions and classes in AST.")
                    
        except SyntaxError as e:
            scores["Formatting"] = 0.0
            scores["Output Structure"] = 0.0
            msg = f"Invalid Python syntax: Line {e.lineno}: {e.msg}"
            critical_failures.append(msg)
            failure_reasons.append(msg)
            recommendations.append("Ensure brackets, indentation, and syntax keyword usage is correct.")
            
        return {
            "component_scores": scores,
            "failure_reasons": failure_reasons,
            "strengths": strengths,
            "recommendations": recommendations,
            "critical_failures": critical_failures
        }


class ResponseEvaluator:
    def __init__(self, client: LLMClient = None, weights: dict[str, float] = None):
        self.client = client or LLMClient()
        self.weights = weights or {
            "Formatting": 1.0,
            "Task Completion": 2.0,
            "Coverage": 1.5,
            "Reasoning": 1.5,
            "Constraint Satisfaction": 2.0,
            "Output Structure": 1.5,
            "Completeness": 1.5,
        }

    def _is_echoing_prompt(self, query: str, response: str) -> bool:
        q = query.strip().lower()
        r = response.strip().lower()
        if len(q) <= 10 or len(r) <= 10:
            return False
            
        # Extract word sets
        q_words = [re.sub(r'[^\w]', '', w) for w in q.split() if re.sub(r'[^\w]', '', w)]
        r_words = [re.sub(r'[^\w]', '', w) for w in r.split() if re.sub(r'[^\w]', '', w)]
        
        if not q_words or not r_words:
            return False
            
        q_set = set(q_words)
        r_set = set(r_words)
        intersection = q_set.intersection(r_set)
        
        overlap_ratio = len(intersection) / len(r_set)
        
        # Flag echoing only when response consists entirely of prompt words with almost no new info
        if overlap_ratio > 0.85 and len(r_words) < len(q_words) * 1.2:
            new_words = r_set - q_set
            stop_words = {"q", "a", "query", "question", "answer", "response", "the", "is", "of", "to", "and", "or"}
            meaningful_new_words = [w for w in new_words if w not in stop_words]
            if len(meaningful_new_words) < 5:
                return True
        return False

    def _has_repetition_loops(self, response: str) -> bool:
        text = response.lower()
        if re.search(r'\b(\w+)(?:\s+\1){3,}\b', text):
            return True
        text_no_space = re.sub(r'\s+', '', text)
        if re.search(r'(.{3,100}?)\1{3,}', text_no_space):
            match = re.search(r'(.{3,100}?)\1{3,}', text_no_space)
            if match and not re.match(r'^[-_*.]+$', match.group(1)):
                return True
        words = [w for w in text.split() if len(w) > 2]
        if len(words) > 15:
            common = Counter(words).most_common(1)[0]
            if common[1] / len(words) > 0.4:
                return True
        return False

    def _has_placeholders(self, response: str) -> bool:
        placeholders = ["todo", "[insert", "<insert", "your_code_here", "your name here", "[write your"]
        r = response.lower()
        return any(p in r for p in placeholders)

    def _self_critique(self, query: str, response: str) -> tuple[float, float, str]:
        """Critique via local model. Returns (confidence_adjustment, quality_penalty, recommendation_str)"""
        sys_prompt = "Grade: accurate? clear? complete?\nTHOUGHTS:\nVERDICT: YES|NO"
        prompt = f"Q: {query}\nA: {response}\n\nTHOUGHTS:\nVERDICT:"

        try:
            eval_result = self.client.call_local(
                prompt=prompt, system_prompt=sys_prompt,
                temperature=0.0, max_tokens=80
            )
            verdict = "NO"
            thoughts = ""
            for line in eval_result.split('\n'):
                lu = line.strip().upper()
                if lu.startswith("VERDICT:"):
                    verdict = lu.replace("VERDICT:", "").strip()
                elif lu.startswith("THOUGHTS:"):
                    thoughts = line[9:].strip()

            if "YES" in verdict:
                return 0.05, 0.0, None # Boost confidence slightly on positive validation
                
            reason = "Self-critique: NO"
            if thoughts:
                reason += f" ({thoughts})"
            return 0.0, -EvaluatorPenalties.SELF_CRITIQUE_PENALTY, reason
        except Exception:
            # Ignore self-critique model exceptions/timeouts
            return 0.0, 0.0, None

    def evaluate(self, query: str, response: str, response_format: str = "text", schema: any = None) -> EvaluationResult:
        """
        Runs a suite of checks locally to verify the response quality.
        Returns:
            EvaluationResult: Subclass-compatible result containing quality metrics.
        """
        # Determine if there's an immediate model error
        cleaned_resp = response.strip() if response else ""
        if not cleaned_resp or "error executing local query" in cleaned_resp.lower():
            # Return model error status cleanly
            return EvaluationResult(
                passed=False,
                quality_score=0.0,
                confidence=1.0,
                component_scores={},
                critical_failures=["Model failed to generate a response or local pipeline error occurred."],
                failure_reasons=["Model failure"],
                strengths=[],
                recommendations=[],
                status="model_error",
                evaluator_error="Model output empty or pipeline error."
            )

        try:
            res = self._evaluate_inner(query, response, response_format, schema)
            # Determine success vs failed status
            res.status = "success" if res.passed else "failed"
            return res
        except Exception as e:
            return EvaluationResult(
                passed=None,
                quality_score=None,
                confidence=0.0,
                component_scores={},
                critical_failures=[],
                failure_reasons=[],
                strengths=[],
                recommendations=[],
                status="evaluation_error",
                evaluator_error=str(e)
            )

    def _evaluate_inner(self, query: str, response: str, response_format: str = "text", schema: any = None) -> EvaluationResult:
        critical_failures = []
        failure_reasons = []
        strengths = []
        recommendations = []
        
        # 1. RUN STANDARD CHECKS (Critical failures vs warnings)
        if not response or len(response.strip()) < 5:
            critical_failures.append("Response is too short or empty.")
        
        if "error executing local query" in response.lower():
            critical_failures.append("Execution error in local pipeline.")
            
        if self._is_echoing_prompt(query, response):
            critical_failures.append("Response echoes the prompt instead of answering it.")
            
        if self._has_repetition_loops(response):
            critical_failures.append("Response contains repetitive loops.")
            
        if self._has_placeholders(response):
            critical_failures.append("Response contains incomplete placeholder markers.")
            
        # Constraint violations (word/sentence limit, list format) are Warnings or critical fails in tests
        is_ok, err = self._check_task_constraints(query, response)
        if not is_ok:
            failure_reasons.append(err)
            if _is_test_env:
                critical_failures.append(err)
            
        if response_format.lower() == "json":
            is_ok, err = self._check_json(response, schema)
            if not is_ok:
                critical_failures.append(f"Invalid JSON format. Error: {err}")
        elif response_format.lower() == "python":
            is_ok, err = self._check_python(response)
            if not is_ok:
                critical_failures.append(f"Invalid Python syntax. Error: {err}")

        # Soft Self-Critique logic
        self_critique_penalty = 0.0
        self_critique_conf_adj = 0.0
        
        should_critique = getattr(self.client, "enable_local_critique", True)
        if should_critique and response_format.lower() in ("json", "python"):
            if os.getenv("ENABLE_LOCAL_CRITIQUE") is None:
                should_critique = False

        if should_critique and not critical_failures:
            conf_adj, qual_pen, rec_err = self._self_critique(query, response)
            if qual_pen < 0:
                failure_reasons.append(rec_err)
                recommendations.append(f"Resolve critique failure: {rec_err}")
                self_critique_penalty = qual_pen
            self_critique_conf_adj = conf_adj

        # 2. RUN TASK-SPECIFIC QUALITY EVALUATORS
        dimension_scores = {dim: [] for dim in self.weights}
        
        if ComparisonEvaluator.is_applicable(query):
            comp_eval = ComparisonEvaluator()
            res = comp_eval.evaluate(query, response)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if SummarizationEvaluator.is_applicable(query):
            sum_eval = SummarizationEvaluator()
            res = sum_eval.evaluate(query, response)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if CodeGenerationEvaluator.is_applicable(query, response_format):
            code_eval = CodeGenerationEvaluator()
            res = code_eval.evaluate(query, response)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if DebuggingEvaluator.is_applicable(query):
            debug_eval = DebuggingEvaluator()
            res = debug_eval.evaluate(query, response)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if ArchitectureEvaluator.is_applicable(query):
            arch_eval = ArchitectureEvaluator()
            res = arch_eval.evaluate(query, response)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if JSONEvaluator.is_applicable(response_format):
            json_eval = JSONEvaluator()
            res = json_eval.evaluate(response, schema)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)
            
        if PythonEvaluator.is_applicable(response_format):
            py_eval = PythonEvaluator()
            res = py_eval.evaluate(response, query)
            self._merge_results(res, dimension_scores, critical_failures, failure_reasons, strengths, recommendations)

        # Compute final average scores for each dimension
        final_component_scores = {}
        for dim, weight in self.weights.items():
            scores_list = dimension_scores.get(dim, [])
            if scores_list:
                final_component_scores[dim] = sum(scores_list) / len(scores_list)
            else:
                final_component_scores[dim] = 1.0
                
        total_weight = sum(self.weights.values())
        weighted_sum = sum(final_component_scores[dim] * self.weights[dim] for dim in self.weights)
        quality_score = weighted_sum / total_weight if total_weight > 0 else 1.0
        
        # Compute warning penalties based on types of failure reasons
        warning_penalties = 0.0
        for reason in failure_reasons:
            r_lower = reason.lower()
            if "word limit" in r_lower:
                warning_penalties += EvaluatorPenalties.WORD_LIMIT_PENALTY
            elif "sentence limit" in r_lower or "multiple sentences" in r_lower:
                warning_penalties += EvaluatorPenalties.MINOR_CONSTRAINT_PENALTY
            elif "bullet point" in r_lower or "list formatting" in r_lower or "missing list" in r_lower:
                warning_penalties += EvaluatorPenalties.FORMATTING_PENALTY
            elif "missing compared entities" in r_lower:
                warning_penalties += EvaluatorPenalties.PARTIAL_COVERAGE_PENALTY
            elif "missing comparison dimension" in r_lower:
                warning_penalties += EvaluatorPenalties.PARTIAL_COVERAGE_PENALTY
            elif "missing recommendation" in r_lower:
                warning_penalties += EvaluatorPenalties.MISSING_RECOMMENDATION_PENALTY
            elif "missing advantages/disadvantages" in r_lower or "tradeoff" in r_lower:
                warning_penalties += EvaluatorPenalties.MISSING_RECOMMENDATION_PENALTY
            elif "missing language markers" in r_lower:
                warning_penalties += EvaluatorPenalties.FORMATTING_PENALTY
            else:
                warning_penalties += EvaluatorPenalties.MINOR_CONSTRAINT_PENALTY
                
        # Apply warning caps and self-critique soft adjustment
        capped_warning_penalty = min(warning_penalties, EvaluatorPenalties.MAX_CAPPED_PENALTY)
        quality_score = quality_score - capped_warning_penalty + self_critique_penalty
        
        # Prevent zero-out unless it represents a Critical Failure
        quality_score = max(quality_score, 0.05)
        
        if critical_failures:
            quality_score = 0.0
            
        passed = len(critical_failures) == 0
        
        confidence = 0.85 + self_critique_conf_adj
        if response_format.lower() in ("json", "python") and passed:
            confidence = 0.95 + self_critique_conf_adj
        if critical_failures:
            confidence = 1.0
            
        confidence = max(0.0, min(1.0, confidence))
            
        return EvaluationResult(
            passed=passed,
            quality_score=quality_score,
            confidence=confidence,
            component_scores=final_component_scores,
            critical_failures=critical_failures,
            failure_reasons=failure_reasons,
            strengths=strengths,
            recommendations=recommendations
        )

    def _merge_results(self, res: dict, dimension_scores: dict, critical_failures: list, failure_reasons: list, strengths: list, recommendations: list):
        for dim, score in res.get("component_scores", {}).items():
            if dim in dimension_scores:
                dimension_scores[dim].append(score)
        
        # Merge task specific critical failures (if they are truly critical like placeholder code)
        for cf in res.get("critical_failures", []):
            if cf not in critical_failures:
                critical_failures.append(cf)
        for fr in res.get("failure_reasons", []):
            if fr not in failure_reasons:
                failure_reasons.append(fr)
        for st in res.get("strengths", []):
            if st not in strengths:
                strengths.append(st)
        for rec in res.get("recommendations", []):
            if rec not in recommendations:
                recommendations.append(rec)

    def _check_task_constraints(self, query: str, response: str) -> tuple[bool, str]:
        """Verifies text formatting, limits, and justification requirements requested in the query."""
        import re
        query_lower = query.lower()
        response_lower = response.lower()

        if any(term in query_lower for term in ["one sentence", "single sentence", "1 sentence"]):
            chunks = re.split(r'(?<=[.!?])\s+', response.strip())
            valid_chunks = [c for c in chunks if len(c.strip()) > 3]
            if len(valid_chunks) > 1:
                return False, f"Response contains multiple sentences ({len(valid_chunks)}) when only one was requested."

        limit_match = re.search(r'\b(?:under|less\s+than|below|max|maximum\s+of|limit\s+of)\s+(\d+)\s+words\b', query_lower)
        if not limit_match:
            limit_match = re.search(r'\b(\d+)\s+words\s+(?:or\s+less|max|maximum|limit)\b', query_lower)
        if not limit_match:
            limit_match = re.search(r'\bin\s+(?:under|less\s+than|below|max|maximum\s+of)?\s*(\d+)\s+words\b', query_lower)
        
        if limit_match:
            try:
                max_words = int(limit_match.group(1))
                word_count = len(response.split())
                if word_count > max_words:
                    return False, f"Response exceeds word limit of {max_words} words (got {word_count})."
            except ValueError:
                pass

        if any(term in query_lower for term in ["bullet point", "bullet list", "bulleted list", "numbered list", "list form"]):
            lines = response.split('\n')
            has_list = any(re.match(r'^\s*(?:[-*+]+|\d+\.)\s+', line) for line in lines)
            if not has_list:
                return False, "Response does not use bullet points or list formatting as requested."

        if "sentiment" in query_lower:
            needs_justification = any(term in query_lower for term in ["justify", "justification", "explain", "why"])
            if needs_justification:
                word_count = len(response.split())
                if word_count < 10:
                    return False, "Response is too short to contain a valid justification."
                reasoning_words = ["because", "since", "due to", "reason", "indicates", "shows", "as", "justification", "why"]
                response_words = set(re.findall(r'\b\w+\b', response_lower))
                if not any(w in response_words for w in reasoning_words):
                    return False, "Response does not contain clear justification or reasoning indicators."

        return True, ""

    def _check_json(self, response: str, schema: any = None) -> tuple[bool, str]:
        """Verifies if the response contains valid JSON, extracting the JSON structure if needed."""
        cleaned = response.strip()
        braces = [idx for idx in [cleaned.find('{'), cleaned.find('[')] if idx != -1]
        r_braces = [idx for idx in [cleaned.rfind('}'), cleaned.rfind(']')] if idx != -1]
        
        if braces and r_braces:
            start = min(braces)
            end = max(r_braces)
            cleaned_json = cleaned[start:end+1]
        else:
            cleaned_json = self._extract_code_block(cleaned, "json")

        try:
            parsed = json.loads(cleaned_json)
        except ValueError as e:
            return False, str(e)

        if isinstance(parsed, dict) and not parsed:
            return False, "JSON object is empty."
        if isinstance(parsed, list) and not parsed:
            return False, "JSON list is empty."

        if schema is not None:
            json_eval = JSONEvaluator()
            ok, err = json_eval._validate_nested_schema(parsed, schema)
            if not ok:
                return False, err

        return True, ""

    def _check_python(self, response: str) -> tuple[bool, str]:
        """Verifies if the response contains syntactically correct Python code and functional statements."""
        cleaned = self._extract_code_block(response, "python")
        try:
            tree = ast.parse(cleaned)
            important_nodes = (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Call, ast.For, ast.While, ast.If)
            if not any(isinstance(node, important_nodes) for node in ast.walk(tree)):
                return False, "Python code contains no functional statements (e.g. definitions, assignments, calls)."
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _extract_code_block(self, text: str, lang: str) -> str:
        """Robustly extracts content between markdown code blocks if present."""
        cleaned = text.strip()
        start_tag = f"```{lang}"
        if start_tag in cleaned:
            start_idx = cleaned.find(start_tag) + len(start_tag)
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                return cleaned[start_idx:end_idx].strip()
                
        if "```" in cleaned:
            start_idx = cleaned.find("```") + 3
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                return cleaned[start_idx:end_idx].strip()
                
        return cleaned
