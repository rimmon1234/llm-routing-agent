import os
import json
import ast
from .client import LLMClient

class ResponseEvaluator:
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()

    def _is_echoing_prompt(self, query: str, response: str) -> bool:
        q = query.strip().lower()
        r = response.strip().lower()
        if len(q) > 10 and q in r and len(r) < len(q) * 1.3:
            return True
        return False

    def _has_repetition_loops(self, response: str) -> bool:
        text = response.lower()
        import re
        if re.search(r'\b(\w+)(?:\s+\1){3,}\b', text):
            return True
        if re.search(r'(.{2,20}?)\1{3,}', text):
            match = re.search(r'(.{2,20}?)\1{3,}', text)
            if match and match.group(1).strip() not in ("", "-", "_", "*", "."):
                return True
        words = [w for w in text.split() if len(w) > 2]
        if len(words) > 15:
            from collections import Counter
            common = Counter(words).most_common(1)[0]
            if common[1] / len(words) > 0.4:
                return True
        return False

    def _has_placeholders(self, response: str) -> bool:
        placeholders = ["todo", "[insert", "<insert", "your_code_here", "your name here", "[write your"]
        r = response.lower()
        return any(p in r for p in placeholders)

    def evaluate(self, query: str, response: str, response_format: str = "text", schema: any = None) -> tuple[bool, str]:
        """
        Runs a suite of checks locally to verify the response quality.
        Returns:
            tuple: (is_valid: bool, error_reason: str)
        """
        # Basic sanity check
        if not response or len(response.strip()) < 5:
            return False, "Response is too short or empty."

        # Error checks
        if "error executing local query" in response.lower():
            return False, "Execution error in local pipeline."

        # Advanced stability and quality checks
        if self._is_echoing_prompt(query, response):
            return False, "Response echoes the prompt instead of answering it."

        if self._has_repetition_loops(response):
            return False, "Response contains repetitive loops."

        if self._has_placeholders(response):
            return False, "Response contains incomplete placeholder markers."

        # Format-specific checks
        if response_format.lower() == "json":
            is_ok, err = self._check_json(response, schema)
            if not is_ok:
                return False, f"Invalid JSON format. Error: {err}"
        elif response_format.lower() == "python":
            is_ok, err = self._check_python(response)
            if not is_ok:
                return False, f"Invalid Python syntax. Error: {err}"

        # LLM Self-Critique / Quality Verification
        # Only run critique if enabled on the client.
        # For structured formats (json, python) that successfully passed structural validation,
        # we bypass self-critique by default unless explicitly forced via env configuration.
        should_critique = getattr(self.client, "enable_local_critique", True)
        if should_critique and response_format.lower() in ("json", "python"):
            if os.getenv("ENABLE_LOCAL_CRITIQUE") is None:
                should_critique = False

        if should_critique:
            is_ok, err = self._self_critique(query, response)
            if not is_ok:
                return False, f"Self-critique failed: {err}"

        return True, ""

    def _check_json(self, response: str, schema: any = None) -> tuple[bool, str]:
        """Verifies if the response contains valid JSON, extracting the JSON structure if needed."""
        cleaned = response.strip()
        
        # Try robust extraction between matching brackets first
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

        # Check for empty JSON structures
        if isinstance(parsed, dict) and not parsed:
            return False, "JSON object is empty."
        if isinstance(parsed, list) and not parsed:
            return False, "JSON list is empty."

        # Apply schema/keys validation if provided
        if schema is not None:
            # 1. Pydantic Model (v2 / v1)
            if hasattr(schema, "model_validate_json"):
                try:
                    schema.model_validate_json(cleaned_json)
                except Exception as e:
                    return False, f"Schema validation error: {str(e)}"
            elif hasattr(schema, "parse_raw"):
                try:
                    schema.parse_raw(cleaned_json)
                except Exception as e:
                    return False, f"Schema validation error: {str(e)}"
            
            # 2. List or set of required keys
            elif isinstance(schema, (list, set)):
                if isinstance(parsed, dict):
                    missing_keys = [k for k in schema if k not in parsed]
                    if missing_keys:
                        return False, f"Missing required keys: {missing_keys}"
                elif isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        if isinstance(item, dict):
                            missing_keys = [k for k in schema if k not in item]
                            if missing_keys:
                                return False, f"Item at index {idx} missing keys: {missing_keys}"
                        else:
                            return False, f"Expected dictionary at index {idx}, got {type(item).__name__}"
                else:
                    return False, f"Expected dictionary or list of dictionaries to validate keys, got {type(parsed).__name__}"

            # 3. Dictionary of key-to-type mappings
            elif isinstance(schema, dict):
                if not isinstance(parsed, dict):
                    return False, f"Expected a JSON object (dictionary), got {type(parsed).__name__}"
                for key, expected_type in schema.items():
                    if key not in parsed:
                        return False, f"Missing required key: '{key}'"
                    val = parsed[key]
                    if isinstance(expected_type, type) or (isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type)):
                        if not isinstance(val, expected_type):
                            return False, f"Key '{key}' expected type {expected_type.__name__}, got {type(val).__name__}"

        return True, ""

    def _check_python(self, response: str) -> tuple[bool, str]:
        """Verifies if the response contains syntactically correct Python code and functional statements."""
        cleaned = self._extract_code_block(response, "python")
        try:
            tree = ast.parse(cleaned)
            # Ensure it is not just comments/docstrings by walking the AST
            important_nodes = (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Call, ast.For, ast.While, ast.If)
            if not any(isinstance(node, important_nodes) for node in ast.walk(tree)):
                return False, "Python code contains no functional statements (e.g. definitions, assignments, calls)."
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _extract_code_block(self, text: str, lang: str) -> str:
        """Robustly extracts content between markdown code blocks if present."""
        cleaned = text.strip()
        
        # Search for ```lang ... ```
        start_tag = f"```{lang}"
        if start_tag in cleaned:
            start_idx = cleaned.find(start_tag) + len(start_tag)
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                return cleaned[start_idx:end_idx].strip()
                
        # Fallback to generic ``` ... ```
        if "```" in cleaned:
            start_idx = cleaned.find("```") + 3
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                return cleaned[start_idx:end_idx].strip()
                
        return cleaned

    def _self_critique(self, query: str, response: str) -> tuple[bool, str]:
        """Critique via local model. Uses compact prompt to minimize token use."""
        sys = "Grade: accurate? clear? complete?\nTHOUGHTS:\nVERDICT: YES|NO"
        prompt = f"Q: {query}\nA: {response}\n\nTHOUGHTS:\nVERDICT:"

        try:
            eval_result = self.client.call_local(
                prompt=prompt, system_prompt=sys,
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
                return True, ""
            reason = "Self-critique: NO"
            if thoughts:
                reason += f" ({thoughts})"
            return False, reason
        except Exception:
            return True, ""
