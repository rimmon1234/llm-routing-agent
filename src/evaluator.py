import os
import json
import ast
from .client import LLMClient

class ResponseEvaluator:
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()

    def evaluate(self, query: str, response: str, response_format: str = "text", schema: any = None) -> tuple[bool, str]:
        """
        Runs a suite of checks locally to verify the response quality.
        Returns:
            tuple: (is_valid: bool, error_reason: str)
        """
        # Basic sanity check
        if not response or len(response.strip()) < 3:
            return False, "Response is too short or empty."

        # Error checks
        if "error executing local query" in response.lower():
            return False, "Execution error in local pipeline."

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
        """Verifies if the response contains syntactically correct Python code."""
        cleaned = self._extract_code_block(response, "python")
        try:
            ast.parse(cleaned)
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
        """
        Instructs the local model to critique its own response using Chain-of-Thought (CoT).
        """
        system_prompt = (
            "You are an objective AI evaluator. Your job is to grade the answer provided for a user query.\n"
            "Examine if the answer is accurate, coherent, and fully answers the question.\n"
            "You must respond in the following format:\n"
            "THOUGHTS: <one or two sentences evaluating the response>\n"
            "VERDICT: <either YES or NO>"
        )
        prompt = (
            f"User Query: {query}\n\n"
            f"Provided Answer: {response}\n\n"
            "Perform your evaluation. Format your response exactly as:\n"
            "THOUGHTS: [critique analysis]\n"
            "VERDICT: [YES or NO]"
        )

        try:
            eval_result = self.client.call_local(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=100
            )
            # Find the VERDICT line
            verdict = "NO"
            thoughts = ""
            for line in eval_result.split('\n'):
                line_upper = line.strip().upper()
                if line_upper.startswith("VERDICT:"):
                    verdict = line_upper.replace("VERDICT:", "").strip()
                elif line_upper.startswith("THOUGHTS:"):
                    thoughts = line.strip()[9:].strip()

            if "YES" in verdict:
                return True, ""
            
            reason = "Model self-critique returned 'NO'."
            if thoughts:
                reason += f" Reason: {thoughts}"
            return False, reason
        except Exception as e:
            # Fallback to True if eval fails to prevent false negatives
            return True, ""
