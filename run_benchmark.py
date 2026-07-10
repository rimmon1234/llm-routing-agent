import os
import sys
import json
import time
import csv
import argparse
import inspect
from datetime import datetime

# Import core routing framework
from src.client import LLMClient
from src.router import HybridRouter
from src.evaluator import ResponseEvaluator, detect_programming_language
from src.style import S

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Global Telemetry Tracker for Non-Invasive Instrumentation
class TelemetryTracker:
    def __init__(self):
        self.active = False
        self.reset()
        
    def reset(self):
        # Latencies
        self.classifier_latency = 0.0
        self.local_model_latency = 0.0
        self.evaluator_latency = 0.0
        self.repair_latency = 0.0
        self.remote_model_latency = 0.0
        
        # Stage-level prompt/completion token accounting
        self.classifier_prompt_tokens = 0
        self.classifier_completion_tokens = 0
        
        self.local_prompt_tokens = 0
        self.local_completion_tokens = 0
        
        self.evaluator_prompt_tokens = 0
        self.evaluator_completion_tokens = 0
        
        self.repair_prompt_tokens = 0
        self.repair_completion_tokens = 0
        
        self.remote_prompt_tokens = 0
        self.remote_completion_tokens = 0
        
        # Helper states
        self.repair_used = False
        self.remote_fallback_occurred = False
        self.local_attempts = 0
        self.remote_attempts = 0
        
        # Cache of last evaluation to avoid double evaluation
        self.last_evaluated_query = None
        self.last_evaluated_response = None
        self.last_evaluation_result = None

tracker = TelemetryTracker()

# Frame inspection helpers to identify caller state without fragile string matching
def is_caller_repair():
    try:
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "_run_local_with_retry":
                if "attempt" in frame.f_locals:
                    return True
            frame = frame.f_back
    except Exception:
        pass
    return False

def is_caller_self_critique():
    try:
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "_self_critique":
                return True
            frame = frame.f_back
    except Exception:
        pass
    return False

# Monkeypatch LLMClient.call_local to intercept usage data directly
original_call_local = LLMClient.call_local
def wrapped_call_local(self, prompt, *args, **kwargs):
    if not tracker.active:
        return original_call_local(self, prompt, *args, **kwargs)
        
    messages = []
    system_prompt = kwargs.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    temperature = kwargs.get("temperature", 0.2)
    max_tokens = kwargs.get("max_tokens", 1000)
    json_mode = kwargs.get("json_mode", False)
    
    create_kwargs = {}
    if json_mode:
        create_kwargs["response_format"] = {"type": "json"}
        
    start = time.perf_counter()
    res_text = ""
    pt_used = 0
    ct_used = 0
    
    try:
        response = self.local_client.chat.completions.create(
            model=self.local_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **create_kwargs
        )
        res_text = response.choices[0].message.content
        if hasattr(response, "usage") and response.usage:
            pt_used = response.usage.prompt_tokens
            ct_used = response.usage.completion_tokens
    except Exception as e:
        res_text = f"Error executing local query: {str(e)}"
        
    elapsed = time.perf_counter() - start
    
    # Fallback to tiktoken estimation consistently if client doesn't return usage
    if pt_used == 0:
        pt_used = self.estimate_tokens(prompt)
        if system_prompt:
            pt_used += self.estimate_tokens(system_prompt)
    if ct_used == 0:
        ct_used = self.estimate_tokens(res_text)
        
    # Attribute tokens and latency to the correct stage
    if is_caller_self_critique():
        tracker.evaluator_prompt_tokens += pt_used
        tracker.evaluator_completion_tokens += ct_used
    elif is_caller_repair():
        tracker.repair_prompt_tokens += pt_used
        tracker.repair_completion_tokens += ct_used
        tracker.repair_latency += elapsed
        tracker.repair_used = True
    else:
        tracker.local_prompt_tokens += pt_used
        tracker.local_completion_tokens += ct_used
        tracker.local_model_latency += elapsed
        
    tracker.local_attempts += 1
    return res_text

# Monkeypatch LLMClient.call_remote to record remote tokens
original_call_remote = LLMClient.call_remote
def wrapped_call_remote(self, prompt, *args, **kwargs):
    if not tracker.active:
        return original_call_remote(self, prompt, *args, **kwargs)
        
    start = time.perf_counter()
    res, pt, ct = original_call_remote(self, prompt, *args, **kwargs)
    elapsed = time.perf_counter() - start
    
    tracker.remote_model_latency += elapsed
    tracker.remote_prompt_tokens += pt
    tracker.remote_completion_tokens += ct
    tracker.remote_attempts += 1
    return res, pt, ct

# Monkeypatch ResponseEvaluator.evaluate with cache to avoid double evaluation
original_evaluate = ResponseEvaluator.evaluate
def wrapped_evaluate(self, query, response, *args, **kwargs):
    if not tracker.active:
        return original_evaluate(self, query, response, *args, **kwargs)
        
    # Check evaluation cache
    if tracker.last_evaluated_query == query and tracker.last_evaluated_response == response:
        return tracker.last_evaluation_result
        
    start = time.perf_counter()
    eval_res = original_evaluate(self, query, response, *args, **kwargs)
    elapsed = time.perf_counter() - start
    
    tracker.evaluator_latency += elapsed
    
    tracker.last_evaluated_query = query
    tracker.last_evaluated_response = response
    tracker.last_evaluation_result = eval_res
    return eval_res

# Apply monkeypatches
LLMClient.call_local = wrapped_call_local
LLMClient.call_remote = wrapped_call_remote
ResponseEvaluator.evaluate = wrapped_evaluate


def get_response_format(category, prompt):
    cat_lower = category.lower()
    prompt_lower = prompt.lower()
    
    # 1. Check for json explicitly first
    if "json" in prompt_lower or cat_lower == "json/xml":
        if "xml" not in prompt_lower or "json" in prompt_lower:
            return "json"
            
    # 2. Detect programming language dynamically
    lang = detect_programming_language(prompt)
    if lang not in ("text", "code"):
        return lang
        
    # 3. Use category mappings if language is generic or not found
    if cat_lower == "coding":
        return lang if lang != "text" else "code"
    if cat_lower == "sql":
        return "sql"
    if cat_lower == "regex":
        return "text"
        
    if lang == "code":
        return "code"
        
    return "text"



def get_selected_evaluator(query: str, response_format: str) -> str:
    fmt = response_format.lower()
    evaluators = []
    
    from src.evaluator import (
        ComparisonEvaluator, SummarizationEvaluator, ArchitectureEvaluator, 
        DebuggingEvaluator, JSONEvaluator, PythonEvaluator, CodeGenerationEvaluator
    )
    
    if ComparisonEvaluator.is_applicable(query):
        evaluators.append("ComparisonEvaluator")
    if SummarizationEvaluator.is_applicable(query):
        evaluators.append("SummarizationEvaluator")
    if ArchitectureEvaluator.is_applicable(query):
        evaluators.append("ArchitectureEvaluator")
    if DebuggingEvaluator.is_applicable(query):
        evaluators.append("DebuggingEvaluator")
    if JSONEvaluator.is_applicable(fmt):
        evaluators.append("JSONEvaluator")
    if PythonEvaluator.is_applicable(fmt):
        evaluators.append("PythonEvaluator")
    if CodeGenerationEvaluator.is_applicable(query, fmt):
        evaluators.append("CodeGenerationEvaluator")
        
    if not evaluators:
        evaluators.append("GenericTextEvaluator")
        
    return "+".join(evaluators)


def normalize_route(route):
    """Normalize routing decisions for comparison."""
    if not route:
        return "none"
    r = route.lower()
    if "local" in r:
        return "local"
    if "remote" in r:
        return "remote"
    return r


def generate_accuracy_svg(accuracies, output_path):
    """Generate a clean side-by-side vertical bar chart SVG for strategy accuracy."""
    width = 500
    height = 300
    margin = 50
    
    strategies = list(accuracies.keys())
    
    # SVG construction
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; border-radius:8px; font-family:system-ui, sans-serif;">',
        '<!-- Title -->',
        '<text x="250" y="30" fill="#cdd6f4" font-size="16" font-weight="bold" text-anchor="middle">Accuracy (Pass Rate) by Strategy</text>',
        '<!-- Grid lines -->'
    ]
    
    for i in range(5):
        y = margin + i * (height - 2 * margin) / 4
        val = 100 - i * 25
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" stroke="#45475a" stroke-dasharray="4"/>')
        svg.append(f'<text x="{margin - 10}" y="{y + 4}" fill="#a6adc8" font-size="11" text-anchor="end">{val}%</text>')
        
    bar_width = 60
    gap = 40
    colors = ["#f38ba8", "#89b4fa", "#f9e2af", "#a6e3a1"]
    
    for i, (strat, val) in enumerate(accuracies.items()):
        x = margin + gap + i * (bar_width + gap)
        bar_height = (val / 100.0) * (height - 2 * margin)
        y = height - margin - bar_height
        color = colors[i % len(colors)]
        
        # Draw bar
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="{color}"/>')
        # Draw value label
        svg.append(f'<text x="{x + bar_width/2}" y="{y - 8}" fill="#cdd6f4" font-size="12" font-weight="bold" text-anchor="middle">{val:.1f}%</text>')
        # Draw X-axis label
        label = strat.replace("always_", "").capitalize()
        svg.append(f'<text x="{x + bar_width/2}" y="{height - margin + 20}" fill="#a6adc8" font-size="11" text-anchor="middle">{label}</text>')
        
    svg.append('</svg>')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def generate_cost_svg(costs, output_path):
    """Generate a clean vertical bar chart SVG for total strategy costs."""
    width = 500
    height = 300
    margin = 50
    
    strategies = list(costs.keys())
    values = list(costs.values())
    max_val = max(values) if max(values) > 0 else 1.0
    
    # SVG construction
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e1e2e; border-radius:8px; font-family:system-ui, sans-serif;">',
        '<!-- Title -->',
        '<text x="250" y="30" fill="#cdd6f4" font-size="16" font-weight="bold" text-anchor="middle">Total API Cost ($)</text>',
        '<!-- Grid lines -->'
    ]
    
    for i in range(5):
        y = margin + i * (height - 2 * margin) / 4
        val = max_val - i * (max_val / 4)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" stroke="#45475a" stroke-dasharray="4"/>')
        svg.append(f'<text x="{margin - 10}" y="{y + 4}" fill="#a6adc8" font-size="11" text-anchor="end">${val:.4f}</text>')
        
    bar_width = 60
    gap = 40
    colors = ["#f38ba8", "#89b4fa", "#f9e2af", "#a6e3a1"]
    
    for i, (strat, val) in enumerate(costs.items()):
        x = margin + gap + i * (bar_width + gap)
        bar_height = (val / max_val) * (height - 2 * margin) if max_val > 0 else 0
        y = height - margin - bar_height
        color = colors[i % len(colors)]
        
        # Draw bar
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="{color}"/>')
        # Draw value label
        svg.append(f'<text x="{x + bar_width/2}" y="{y - 8}" fill="#cdd6f4" font-size="12" font-weight="bold" text-anchor="middle">${val:.4f}</text>')
        # Draw X-axis label
        label = strat.replace("always_", "").capitalize()
        svg.append(f'<text x="{x + bar_width/2}" y="{height - margin + 20}" fill="#a6adc8" font-size="11" text-anchor="middle">{label}</text>')
        
    svg.append('</svg>')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def run_benchmark():
    parser = argparse.ArgumentParser(description="Hybrid LLM Router Benchmark Runner")
    parser.add_argument("--limit", type=int, help="Limit execution to the first N prompts for faster debugging")
    parser.add_argument("--strategies", type=str, help="Comma-separated list of strategies to evaluate (always_local, always_remote, fallback, predictive)")
    parser.add_argument("--dry-run-calibration", action="store_true", help="Perform threshold calibration dry-run without writing to calibrated_thresholds.json")
    parser.add_argument("--min-calibration-prompts", type=int, default=50, help="Minimum number of prompts required to persist calibrated thresholds")
    parser.add_argument("--min-improvement", type=float, default=0.02, help="Minimum improvement in oracle routing accuracy to persist calibration")
    parser.add_argument("--latency-tolerance", type=float, default=0.10, help="Allowable latency increase factor (e.g. 0.10 for 10 percent)")
    parser.add_argument("--calibration-step", type=float, default=0.02, help="Grid search step size for threshold calibration")
    parser.add_argument("--benchmark-version", type=str, default="1.1", help="Version identifier for the benchmark dataset")
    args = parser.parse_args()
    
    if args.limit is not None and args.limit < 1:
        print(S.error("Error: --limit must be an integer greater than or equal to 1."))
        sys.exit(1)
        
    all_supported_strategies = ["always_local", "always_remote", "fallback", "predictive"]
    if args.strategies:
        strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
        for s in strategies:
            if s not in all_supported_strategies:
                print(S.error(f"Error: Strategy '{s}' is not supported. Choose from {all_supported_strategies}"))
                sys.exit(1)
    else:
        strategies = all_supported_strategies
        
    dataset_path = "input/benchmark_dataset.json"
    if not os.path.exists(dataset_path):
        print(S.error(f"Error: Dataset {dataset_path} not found. Run scripts/generate_dataset.py first."))
        sys.exit(1)
        
    print(S.header("\n  Initializing LLM Client & Hybrid Router..."))
    client = LLMClient()
    router = HybridRouter(client=client)
    evaluator = router.evaluator
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
        
    total_dataset_size = len(full_dataset)
    executed_size = total_dataset_size
    skipped_size = 0
    
    if args.limit is not None:
        executed_size = min(args.limit, total_dataset_size)
        skipped_size = total_dataset_size - executed_size
        dataset = full_dataset[:executed_size]
        print(S.warn(f"⚠️  Fast Development Mode: Limiting benchmark execution to first {executed_size} prompts."))
    else:
        dataset = full_dataset
        
    results_by_strategy = {strat: [] for strat in strategies}
    
    print(S.header(f"\n  Starting Benchmark Run ({executed_size} prompts across {len(strategies)} strategies)..."))
    
    total_runs = len(strategies) * executed_size
    run_idx = 0
    start_time_all = time.perf_counter()
    
    # Execute loop
    for prompt_idx, item in enumerate(dataset, 1):
        query = item["prompt"]
        category = item["category"]
        difficulty = item["difficulty"]
        id_str = item["id"]
        
        fmt = get_response_format(category, query)
        
        # Pre-calculate complexity report (Routing Overhead)
        class_start = time.perf_counter()
        report = router.analyze_complexity(query, response_format=fmt)
        class_elapsed = time.perf_counter() - class_start
        
        for strategy in strategies:
            run_idx += 1
            elapsed_all = time.perf_counter() - start_time_all
            avg_time_per_run = elapsed_all / run_idx if run_idx > 1 else 0
            remaining_runs = total_runs - run_idx
            eta = remaining_runs * avg_time_per_run
            
            # Print live progress
            sys.stdout.write(f"\r\033[K[{run_idx}/{total_runs}] {strategy.upper()} | {category} | {difficulty.capitalize()} | Elapsed: {elapsed_all:.1f}s | ETA: {eta:.1f}s")
            sys.stdout.flush()
            
            # Reset tracker for non-invasive metrics
            tracker.reset()
            tracker.active = True
            
            # Run routing system
            try:
                res = router.route_and_execute(
                    query, 
                    strategy=strategy, 
                    response_format=fmt,
                    no_cache=False
                )
                final_response = res.get("response", "")
            except Exception as e:
                res = {
                    "query": query,
                    "strategy": strategy,
                    "route_chosen": "error",
                    "response": f"Error: {e}",
                    "cost_dollars": 0.0,
                    "latency_sec": 0.0
                }
                final_response = f"Error: {e}"
                
            tracker.active = False
            
            # Evaluate final response (wrapped_evaluate uses cache internally to prevent double validation)
            try:
                eval_res = evaluator.evaluate(query, final_response, response_format=fmt)
                eval_score = eval_res.quality_score
                eval_passed = eval_res.passed
                eval_criticals = eval_res.critical_failures
                eval_status = getattr(eval_res, "status", "success")
                eval_err_msg = getattr(eval_res, "evaluator_error", None)
            except Exception as e:
                eval_score = None
                eval_passed = None
                eval_criticals = [f"Runner caught evaluator crash: {e}"]
                eval_status = "evaluation_error"
                eval_err_msg = str(e)
                 
            # Language Diagnostics printing
            detected_lang = detect_programming_language(query)
            selected_eval = get_selected_evaluator(query, fmt)
            is_coding = (category.lower() == "coding" or fmt in ("python", "javascript", "java", "cpp", "go", "rust", "typescript", "code", "sql", "bash", "c"))
            if is_coding:
                sys.stdout.write("\n")
                print(f"   [Language Diagnostics - {id_str} | {strategy.upper()}]")
                print(f"      - Detected programming language: {detected_lang}")
                print(f"      - Response format:               {fmt}")
                print(f"      - Selected evaluator:            {selected_eval}")
                print(f"      - Route chosen:                  {normalize_route(res.get('route_chosen'))}")
                print(f"      - Complexity score:              {report.normalized_score:.2f}")
                print(f"      - Risk score:                    {report.risk_score:.2f}")
                print()
                
            # If the response was served from cache, recover token telemetry from the cache object
            if res.get("cached", False):
                route_norm = normalize_route(res.get("route_chosen"))
                if route_norm == "local":
                    tracker.local_prompt_tokens = res.get("prompt_tokens_local", 0)
                    tracker.local_completion_tokens = res.get("completion_tokens_local", 0)
                elif route_norm == "remote":
                    tracker.remote_prompt_tokens = res.get("prompt_tokens_remote", 0)
                    tracker.remote_completion_tokens = res.get("completion_tokens_remote", 0)
            
            # Stage-level token sums
            total_prompt_tokens = (
                tracker.classifier_prompt_tokens +
                tracker.local_prompt_tokens +
                tracker.evaluator_prompt_tokens +
                tracker.repair_prompt_tokens +
                tracker.remote_prompt_tokens
            )
            total_completion_tokens = (
                tracker.classifier_completion_tokens +
                tracker.local_completion_tokens +
                tracker.evaluator_completion_tokens +
                tracker.repair_completion_tokens +
                tracker.remote_completion_tokens
            )
            total_tokens = total_prompt_tokens + total_completion_tokens
            
            # Latency breakdown (ensure routing_overhead maps to classifier_latency)
            routing_overhead_sec = class_elapsed if strategy in ("predictive",) else 0.0
            local_model_latency_sec = tracker.local_model_latency
            remote_model_latency_sec = tracker.remote_model_latency
            repair_latency_sec = tracker.repair_latency
            evaluation_latency_sec = tracker.evaluator_latency
            
            # Model latency is prompt inference + generation
            model_latency_sec = local_model_latency_sec + remote_model_latency_sec + repair_latency_sec
            
            # Total latency calculation (cached retrieves directly)
            if res.get("cached", False):
                total_latency_sec = res.get("latency_sec", 0.0)
            else:
                total_latency_sec = routing_overhead_sec + model_latency_sec + evaluation_latency_sec
                
            record = {
                "id": id_str,
                "prompt": query,
                "category": category,
                "detected_programming_language": detected_lang,
                "selected_evaluator": selected_eval,
                "difficulty": difficulty,
                "reasoning_level": item["reasoning_level"],
                "estimated_complexity": item["estimated_complexity"],
                "strategy": strategy,
                "route_chosen": res.get("route_chosen"),
                "route_chosen_normalized": normalize_route(res.get("route_chosen")),
                "chosen_model": client.local_model if "local" in normalize_route(res.get("route_chosen")) else client.remote_model,
                "routing_confidence": report.confidence,
                "complexity_score": report.normalized_score,
                "risk_score": report.risk_score,
                
                # Detailed Stage-level Latencies
                "routing_overhead_sec": routing_overhead_sec,
                "local_model_latency_sec": local_model_latency_sec,
                "remote_model_latency_sec": remote_model_latency_sec,
                "repair_latency_sec": repair_latency_sec,
                "evaluation_latency_sec": evaluation_latency_sec,
                "model_latency_sec": model_latency_sec,
                "total_latency_sec": total_latency_sec,
                
                # Detailed Stage-level Tokens
                "classifier_prompt_tokens": tracker.classifier_prompt_tokens,
                "classifier_completion_tokens": tracker.classifier_completion_tokens,
                "local_prompt_tokens": tracker.local_prompt_tokens,
                "local_completion_tokens": tracker.local_completion_tokens,
                "evaluator_prompt_tokens": tracker.evaluator_prompt_tokens,
                "evaluator_completion_tokens": tracker.evaluator_completion_tokens,
                "repair_prompt_tokens": tracker.repair_prompt_tokens,
                "repair_completion_tokens": tracker.repair_completion_tokens,
                "remote_prompt_tokens": tracker.remote_prompt_tokens,
                "remote_completion_tokens": tracker.remote_completion_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                
                # Response and evaluation
                "response": final_response,
                "evaluator_score": eval_score,
                "evaluator_passed": eval_passed,
                "evaluator_status": eval_status,
                "evaluator_error": eval_err_msg,
                "repair_used": tracker.repair_used,
                "remote_fallback_occurred": res.get("fallback_triggered", False),
                "cost_dollars": res.get("cost_dollars", 0.0),
                "errors": ", ".join(eval_criticals) if eval_criticals else "",
                "diagnostics_explanation": report.diagnostics.explanation if (report and report.diagnostics) else "",
                "diagnostics_feature_scores": report.diagnostics.feature_scores if (report and report.diagnostics) else {},
                "feature_contributions_version": report.diagnostics.feature_contributions_version if (report and report.diagnostics) else 1,
                "feature_contributions": report.diagnostics.feature_contributions if (report and report.diagnostics) else {},
                
                # Backward Compatibility Mappings
                "total_latency": total_latency_sec,
                "classifier_latency": routing_overhead_sec,
                "evaluator_latency": evaluation_latency_sec,
                "repair_latency": repair_latency_sec,
                "input_tokens": total_prompt_tokens,
                "output_tokens": total_completion_tokens
            }
            results_by_strategy[strategy].append(record)
            time.sleep(0.01) # cooling sleep
            
    print()
    print(S.good("\n  Execution complete. Establishing Cost-Optimal Routes & Normalizing routing labels..."))
    
    # Establish Cost-Optimal Routes (ignoring unsolvable prompts)
    cost_optimal_routes = {}
    local_opts = 0
    remote_opts = 0
    no_opts = 0
    
    for i in range(executed_size):
        prompt_id = dataset[i]["id"]
        query = dataset[i]["prompt"]
        cat = dataset[i]["category"]
        fmt = get_response_format(cat, query)
        
        # Determine local_pass
        local_pass = False
        if "always_local" in results_by_strategy:
            local_pass = results_by_strategy["always_local"][i]["evaluator_passed"]
        else:
            cached = router.cache.get(query, "always_local", fmt)
            if cached:
                try:
                    eval_res = evaluator.evaluate(query, cached["response"], response_format=fmt)
                    local_pass = eval_res.passed
                except Exception:
                    local_pass = False
            else:
                try:
                    res = router.route_and_execute(query, strategy="always_local", response_format=fmt, no_cache=True)
                    eval_res = evaluator.evaluate(query, res.get("response", ""), response_format=fmt)
                    local_pass = eval_res.passed
                    router.cache.set(query, "always_local", fmt, None, res)
                except Exception:
                    local_pass = False
                    
        # Determine remote_pass
        remote_pass = False
        if "always_remote" in results_by_strategy:
            remote_pass = results_by_strategy["always_remote"][i]["evaluator_passed"]
        else:
            cached = router.cache.get(query, "always_remote", fmt)
            if cached:
                try:
                    eval_res = evaluator.evaluate(query, cached["response"], response_format=fmt)
                    remote_pass = eval_res.passed
                except Exception:
                    remote_pass = False
            else:
                try:
                    res = router.route_and_execute(query, strategy="always_remote", response_format=fmt, no_cache=True)
                    eval_res = evaluator.evaluate(query, res.get("response", ""), response_format=fmt)
                    remote_pass = eval_res.passed
                    router.cache.set(query, "always_remote", fmt, None, res)
                except Exception:
                    remote_pass = False
        
        # Oracle route evaluation (Cost-Optimal Route)
        if local_pass and remote_pass:
            gt_route = "local"
            local_opts += 1
        elif not local_pass and remote_pass:
            gt_route = "remote"
            remote_opts += 1
        elif local_pass and not remote_pass:
            gt_route = "local"
            local_opts += 1
        else:
            gt_route = "none" # Unsolvable
            no_opts += 1
            
        cost_optimal_routes[prompt_id] = gt_route
        
    # Annotate records with Cost-Optimal Route comparison
    for strategy in strategies:
        for rec in results_by_strategy[strategy]:
            rec["cost_optimal_route"] = cost_optimal_routes[rec["id"]]
            rec["ground_truth_route"] = cost_optimal_routes[rec["id"]] # backward compatibility
            
            norm_route = rec["route_chosen_normalized"]
            rec["routing_match"] = (norm_route == rec["cost_optimal_route"])
            
    # Calculate aggregates for side-by-side comparison
    aggregates = {}
    for strategy in strategies:
        recs = results_by_strategy[strategy]
        total_p = len(recs)
        eval_errors = sum(1 for r in recs if r.get("evaluator_status") == "evaluation_error")
        model_errors = sum(1 for r in recs if r.get("evaluator_status") == "model_error")
        valid_evaluations = total_p - eval_errors - model_errors
        
        passed_p = sum(1 for r in recs if r.get("evaluator_passed") is True)
        accuracy = (passed_p / valid_evaluations * 100) if valid_evaluations > 0 else 0.0
        
        avg_score = sum(r["evaluator_score"] for r in recs if r.get("evaluator_score") is not None) / valid_evaluations if valid_evaluations > 0 else 0.0
        avg_overhead = sum(r["routing_overhead_sec"] for r in recs) / total_p if total_p > 0 else 0.0
        avg_model_lat = sum(r["model_latency_sec"] for r in recs) / total_p if total_p > 0 else 0.0
        avg_eval_lat = sum(r["evaluation_latency_sec"] for r in recs) / total_p if total_p > 0 else 0.0
        avg_total_lat = sum(r["total_latency_sec"] for r in recs) / total_p if total_p > 0 else 0.0
        
        avg_tokens = sum(r["total_tokens"] for r in recs) / total_p if total_p > 0 else 0.0
        total_cost = sum(r["cost_dollars"] for r in recs)
        avg_cost = total_cost / total_p
        
        local_routed = sum(1 for r in recs if r["route_chosen_normalized"] == "local")
        remote_routed = sum(1 for r in recs if r["route_chosen_normalized"] == "remote")
        
        cost_per_success = total_cost / passed_p if passed_p > 0 else 0.0
        
        aggregates[strategy] = {
            "accuracy": accuracy,
            "avg_score": avg_score,
            "avg_overhead": avg_overhead,
            "avg_model_latency": avg_model_lat,
            "avg_eval_latency": avg_eval_lat,
            "avg_latency": avg_total_lat, # maps to total avg latency
            "avg_tokens": avg_tokens,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "cost_per_success": cost_per_success,
            "local_percent": (local_routed / total_p * 100) if total_p > 0 else 0,
            "remote_percent": (remote_routed / total_p * 100) if total_p > 0 else 0,
            "repair_percent": (sum(1 for r in recs if r["repair_used"]) / total_p * 100) if total_p > 0 else 0,
            "fallback_percent": (sum(1 for r in recs if r["remote_fallback_occurred"]) / total_p * 100) if total_p > 0 else 0,
            "passed_count": passed_p,
            "failed_count": sum(1 for r in recs if r.get("evaluator_status") == "failed"),
            "eval_error_count": eval_errors,
            "model_error_count": model_errors,
        }
        
    # Calculate Oracle Routing Baseline metrics (excluding unsolvable prompts where gt is "none")
    oracle_tokens = 0
    oracle_cost = 0.0
    solvable_count = 0
    
    for i in range(executed_size):
        id_str = dataset[i]["id"]
        gt = cost_optimal_routes[id_str]
        
        # Exclude unsolvable prompts from routing accuracy and routing efficiency sums
        if gt == "none":
            continue
            
        solvable_count += 1
        query = dataset[i]["prompt"]
        cat = dataset[i]["category"]
        fmt = get_response_format(cat, query)
        
        # Get always_local stats
        local_tot = 0
        local_cst = 0.0
        if "always_local" in results_by_strategy:
            local_tot = results_by_strategy["always_local"][i]["total_tokens"]
            local_cst = results_by_strategy["always_local"][i]["cost_dollars"]
        else:
            cached = router.cache.get(query, "always_local", fmt)
            if cached:
                local_tot = client.estimate_tokens(query) + client.estimate_tokens(cached["response"])
                local_cst = 0.0
                
        # Get always_remote stats
        remote_tot = 0
        remote_cst = 0.0
        if "always_remote" in results_by_strategy:
            remote_tot = results_by_strategy["always_remote"][i]["total_tokens"]
            remote_cst = results_by_strategy["always_remote"][i]["cost_dollars"]
        else:
            cached = router.cache.get(query, "always_remote", fmt)
            if cached:
                remote_tot = cached.get("prompt_tokens_remote", 0) + cached.get("completion_tokens_remote", 0)
                remote_cst = cached.get("cost_dollars", 0.0)
                if remote_tot == 0:
                    remote_tot = client.estimate_tokens(query) + client.estimate_tokens(cached["response"])
                    remote_cst = (remote_tot / 1_000_000.0) * client.remote_price_per_1m_tokens
                    
        if gt == "local":
            oracle_tokens += local_tot
            oracle_cost += local_cst
        elif gt == "remote":
            oracle_tokens += remote_tot
            oracle_cost += remote_cst
            
    # Calculate efficiencies only over comparable (solvable) prompts
    pred_total_tokens = 0
    pred_total_cost = 0.0
    if "predictive" in results_by_strategy:
        for i in range(executed_size):
            if cost_optimal_routes[dataset[i]["id"]] != "none":
                pred_total_tokens += results_by_strategy["predictive"][i]["total_tokens"]
                pred_total_cost += results_by_strategy["predictive"][i]["cost_dollars"]
                
    routing_efficiency = (oracle_tokens / pred_total_tokens) if pred_total_tokens > 0 else 0.0
    token_efficiency_pct = routing_efficiency * 100
    cost_ratio = (pred_total_cost / oracle_cost) if oracle_cost > 0 else 1.0
    cost_efficiency_pct = (1.0 / cost_ratio * 100) if cost_ratio > 0 else 100.0
    
    # Calculate Predictive Routing Accuracy (only on comparable/solvable prompts)
    if "predictive" in results_by_strategy:
        comparable_prompts = [r for r in results_by_strategy["predictive"] if r["cost_optimal_route"] in ("local", "remote")]
        if comparable_prompts:
            routing_matches = sum(1 for r in comparable_prompts if r["routing_match"])
            pred_routing_accuracy = (routing_matches / len(comparable_prompts)) * 100
        else:
            pred_routing_accuracy = 0.0
    else:
        pred_routing_accuracy = 0.0
        
    # Calculate Over-Routing and Under-Routing Ratios
    over_routed_count = 0
    under_routed_count = 0
    if "predictive" in results_by_strategy:
        for r in results_by_strategy["predictive"]:
            norm_chosen = r["route_chosen_normalized"]
            gt = r["cost_optimal_route"]
            if norm_chosen == "remote" and gt in ("local", "none"):
                over_routed_count += 1
            elif norm_chosen == "local" and gt == "remote":
                under_routed_count += 1
                
    over_routing_ratio = (over_routed_count / executed_size) * 100
    under_routing_ratio = (under_routed_count / executed_size) * 100
    
    # Estimated savings compared to always-remote
    remote_agg = aggregates.get("always_remote")
    for strategy in strategies:
        agg = aggregates[strategy]
        if remote_agg:
            token_savings_pct = 100 * (remote_agg["avg_tokens"] - agg["avg_tokens"]) / remote_agg["avg_tokens"] if remote_agg["avg_tokens"] > 0 else 0.0
            cost_savings_pct = 100 * (remote_agg["total_cost"] - agg["total_cost"]) / remote_agg["total_cost"] if remote_agg["total_cost"] > 0 else 0.0
            cost_savings_dlr = remote_agg["total_cost"] - agg["total_cost"]
        else:
            token_savings_pct = 0.0
            cost_savings_pct = 0.0
            cost_savings_dlr = 0.0
            
        agg["token_savings_pct"] = max(token_savings_pct, 0.0)
        agg["cost_savings_pct"] = max(cost_savings_pct, 0.0)
        agg["cost_savings_dlr"] = max(cost_savings_dlr, 0.0)
        
    # Per-Category Routing Distribution
    category_metrics = {}
    if "predictive" in results_by_strategy:
        for r in results_by_strategy["predictive"]:
            cat = r["category"]
            if cat not in category_metrics:
                category_metrics[cat] = {
                    "count": 0, "passed": 0, "latency": 0.0, "tokens": 0, "cost": 0.0,
                    "local_count": 0, "remote_count": 0, "repair_count": 0, "fallback_count": 0
                }
            cat_agg = category_metrics[cat]
            cat_agg["count"] += 1
            if r["evaluator_passed"]:
                cat_agg["passed"] += 1
            cat_agg["latency"] += r["total_latency_sec"]
            cat_agg["tokens"] += r["total_tokens"]
            cat_agg["cost"] += r["cost_dollars"]
            
            if r["route_chosen_normalized"] == "local":
                cat_agg["local_count"] += 1
            elif r["route_chosen_normalized"] == "remote":
                cat_agg["remote_count"] += 1
                
            if r["repair_used"]:
                cat_agg["repair_count"] += 1
            if r["remote_fallback_occurred"]:
                cat_agg["fallback_count"] += 1
                
        for cat in category_metrics:
            m = category_metrics[cat]
            c = m["count"]
            m["accuracy"] = (m["passed"] / c * 100) if c > 0 else 0.0
            m["avg_latency"] = m["latency"] / c
            m["avg_tokens"] = m["tokens"] / c
            m["local_pct"] = (m["local_count"] / c * 100) if c > 0 else 0.0
            m["remote_pct"] = (m["remote_count"] / c * 100) if c > 0 else 0.0
            m["repair_pct"] = (m["repair_count"] / c * 100) if c > 0 else 0.0
            m["fallback_pct"] = (m["fallback_count"] / c * 100) if c > 0 else 0.0

    # Difficulty Breakdown
    difficulty_metrics = {}
    if "predictive" in results_by_strategy:
        for r in results_by_strategy["predictive"]:
            diff = r["difficulty"]
            if diff not in difficulty_metrics:
                difficulty_metrics[diff] = {"count": 0, "passed": 0, "tokens": 0, "cost": 0.0}
            difficulty_metrics[diff]["count"] += 1
            if r["evaluator_passed"]:
                difficulty_metrics[diff]["passed"] += 1
            difficulty_metrics[diff]["tokens"] += r["total_tokens"]
            difficulty_metrics[diff]["cost"] += r["cost_dollars"]
            
        for diff in difficulty_metrics:
            m = difficulty_metrics[diff]
            m["accuracy"] = (m["passed"] / m["count"] * 100) if m["count"] > 0 else 0.0
            m["avg_tokens"] = m["tokens"] / m["count"]

    # Sorting Top 20 lists according to correctness rules
    pred_recs = results_by_strategy.get("predictive", [])
    
    # 1. Routing Mistakes sorted by evaluator score ascending (worst first), then cost wasted descending
    routing_mistakes = [r for r in pred_recs if not r["routing_match"] and r["cost_optimal_route"] != "none"]
    routing_mistakes = sorted(routing_mistakes, key=lambda x: (x["evaluator_score"], -x["cost_dollars"]))[:20]
    
    highest_token_prompts = sorted(pred_recs, key=lambda x: x["total_tokens"], reverse=True)[:20]
    
    unnecessary_remote = [r for r in pred_recs if r["route_chosen_normalized"] == "remote" and r["cost_optimal_route"] == "local"]
    unnecessary_remote = sorted(unnecessary_remote, key=lambda x: x["total_tokens"], reverse=True)[:20]
    
    local_failures = [r for r in pred_recs if r["route_chosen_normalized"] == "local" and not r["evaluator_passed"]]
    local_failures = sorted(local_failures, key=lambda x: x["complexity_score"], reverse=True)[:20]
    
    # 2. Evaluator Mistakes: check for true error indicators in passed responses and self-critique conflicts
    eval_discrepancies = []
    for r in pred_recs:
        resp_str = r.get("response") or ""
        has_error_msg = "error executing" in resp_str.lower() or "traceback (most recent call" in resp_str.lower()
        is_suspicious_pass = r["evaluator_passed"] and has_error_msg
        is_critique_conflict = (not r["evaluator_passed"]) and (r["evaluator_score"] > 0.8) and (r["evaluator_tokens"] > 0)
        
        if is_suspicious_pass or is_critique_conflict:
            r["suspicion_reason"] = "Suspicious Pass (Error Msg)" if is_suspicious_pass else "Critique Conflict (High Score, Fail)"
            eval_discrepancies.append(r)
    evaluator_mistakes = sorted(eval_discrepancies, key=lambda x: x["evaluator_score"], reverse=True)[:20]

    # Generate SVGs
    generate_accuracy_svg({s: aggregates[s]["accuracy"] for s in strategies}, "output/accuracy_chart.svg")
    generate_cost_svg({s: aggregates[s]["total_cost"] for s in strategies}, "output/cost_chart.svg")
    
    # Write output/benchmark_results.csv (including granular token splits and backward compatibility columns)
    csv_path = "output/benchmark_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# Benchmark Dataset: {total_dataset_size} prompts\n")
        f.write(f"# Executed: {executed_size}\n")
        f.write(f"# Skipped: {skipped_size}\n")
        f.write(f"# Status: {'Partial Run' if args.limit else 'Full Run'}\n")
        
        writer = csv.writer(f)
        headers = [
            "id", "prompt", "category", "difficulty", "reasoning_level", "estimated_complexity",
            "strategy", "route_chosen", "route_chosen_normalized", "chosen_model", "cost_optimal_route", "routing_match",
            "routing_confidence", "complexity_score", "risk_score",
            
            # Latencies
            "routing_overhead_sec", "local_model_latency_sec", "remote_model_latency_sec", "repair_latency_sec", "evaluation_latency_sec", "model_latency_sec", "total_latency_sec",
            
            # Tokens
            "classifier_prompt_tokens", "classifier_completion_tokens",
            "local_prompt_tokens", "local_completion_tokens",
            "evaluator_prompt_tokens", "evaluator_completion_tokens",
            "repair_prompt_tokens", "repair_completion_tokens",
            "remote_prompt_tokens", "remote_completion_tokens",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            
            # Score / Metadata
            "evaluator_score", "evaluator_passed", "repair_used", "remote_fallback_occurred", "cost_dollars", "errors",
            
            # Backward Compatibility Headers
            "ground_truth_route", "total_latency", "classifier_latency", "evaluator_latency", "repair_latency", "input_tokens", "output_tokens"
        ]
        writer.writerow(headers)
        
        for strategy in strategies:
            for r in results_by_strategy[strategy]:
                writer.writerow([
                    r["id"], r["prompt"], r["category"], r["difficulty"], r["reasoning_level"], r["estimated_complexity"],
                    r["strategy"], r["route_chosen"], r["route_chosen_normalized"], r["chosen_model"], r["cost_optimal_route"], r["routing_match"],
                    r["routing_confidence"], r["complexity_score"], r["risk_score"],
                    
                    r["routing_overhead_sec"], r["local_model_latency_sec"], r["remote_model_latency_sec"], r["repair_latency_sec"], r["evaluation_latency_sec"], r["model_latency_sec"], r["total_latency_sec"],
                    
                    r["classifier_prompt_tokens"], r["classifier_completion_tokens"],
                    r["local_prompt_tokens"], r["local_completion_tokens"],
                    r["evaluator_prompt_tokens"], r["evaluator_completion_tokens"],
                    r["repair_prompt_tokens"], r["repair_completion_tokens"],
                    r["remote_prompt_tokens"], r["remote_completion_tokens"],
                    r["total_prompt_tokens"], r["total_completion_tokens"], r["total_tokens"],
                    
                    r["evaluator_score"], r["evaluator_passed"], r["repair_used"], r["remote_fallback_occurred"], r["cost_dollars"], r["errors"],
                    
                    # Backward Compatibility Values
                    r["cost_optimal_route"], r["total_latency_sec"], r["routing_overhead_sec"], r["evaluation_latency_sec"], r["repair_latency_sec"], r["total_prompt_tokens"], r["total_completion_tokens"]
                ])
                
    # Write output/benchmark_responses.json
    responses_path = "output/benchmark_responses.json"
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(results_by_strategy, f, indent=2, ensure_ascii=False)
        
    # Write routing_analysis.csv
    for path in ["routing_analysis.csv", "output/routing_analysis.csv"]:
        try:
            with open(path, "w", encoding="utf-8", newline="") as f_an:
                writer_an = csv.writer(f_an)
                writer_an.writerow([
                    "prompt", "complexity_score", "feature_contributions", 
                    "predicted_route", "actual_quality_score", "evaluation_result", 
                    "latency", "token_count"
                ])
                target_strategy = "predictive" if "predictive" in results_by_strategy else (strategies[0] if strategies else "")
                if target_strategy and target_strategy in results_by_strategy:
                    for r in results_by_strategy[target_strategy]:
                        writer_an.writerow([
                            r["prompt"],
                            r["complexity_score"],
                            json.dumps(r["diagnostics_feature_scores"]),
                            r["route_chosen_normalized"],
                            r["evaluator_score"],
                            "PASS" if r["evaluator_passed"] else "FAIL",
                            r["total_latency_sec"],
                            r["total_tokens"]
                        ])
            print(S.good(f"[OK] Routing analysis exported to: {path}"))
        except Exception as e:
            print(S.warn(f"Could not write routing analysis to {path}: {e}"))

    # Write output/benchmark_report.md
    report_path = "output/benchmark_report.md"
    report = []
    report.append("# Hybrid LLM Routing Agent Benchmark Report\n")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if args.limit:
        report.append("> [!WARNING]")
        report.append(f"> **Partial Benchmark Run (Fast Development Mode)**: Executing only {executed_size} of {total_dataset_size} prompts.\n")
        
    report.append("## Executive Summary\n")
    report.append(f"- **Benchmark Dataset size**: {total_dataset_size} prompts")
    report.append(f"- **Executed prompts**: {executed_size}")
    report.append(f"- **Skipped prompts**: {skipped_size}")
    
    pred_agg = aggregates.get("predictive", {"accuracy": 0.0, "avg_latency": 0.0, "token_savings_pct": 0.0, "cost_savings_pct": 0.0, "cost_savings_dlr": 0.0})
    fallback_agg = aggregates.get("fallback", {"avg_latency": 0.0})
    local_agg = aggregates.get("always_local", {"accuracy": 0.0})
    remote_agg = aggregates.get("always_remote", {"accuracy": 0.0})
    
    report.append(f"- **Accuracy Gate Pass Rate**: `{pred_agg['accuracy']:.1f}%` (vs. Local `{local_agg['accuracy']:.1f}%`, Remote `{remote_agg['accuracy']:.1f}%`)")
    report.append(f"- **Avg Latency (Predictive)**: `{pred_agg['avg_latency']:.3f}s` (vs. Fallback `{fallback_agg['avg_latency']:.3f}s`)")
    report.append(f"- **Estimated Token Savings**: `{pred_agg['token_savings_pct']:.1f}%` (compared to remote)")
    report.append(f"- **Estimated Cost Savings**: `{pred_agg['cost_savings_pct']:.1f}%` (`${pred_agg['cost_savings_dlr']:.4f}` saved)\n")
    
    report.append("## Overall Metrics Side-by-Side\n")
    report.append("| Strategy | Accuracy | Passed | Failed | Eval Errors | Model Errors | Avg Score | Avg Routing Overhead | Avg Model Latency | Avg Eval Latency | Avg Total Latency | Avg Tokens | Total Cost | Cost per Success | Token Savings vs Remote | Local Routing % | Remote Routing % | Fallback % |")
    report.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for strat in strategies:
        agg = aggregates[strat]
        report.append(f"| `{strat.replace('always_', '').capitalize()}` | {agg['accuracy']:.1f}% | {agg['passed_count']} | {agg['failed_count']} | {agg['eval_error_count']} | {agg['model_error_count']} | {agg['avg_score']:.2f} | {agg['avg_overhead']:.3f}s | {agg['avg_model_latency']:.3f}s | {agg['avg_eval_latency']:.3f}s | {agg['avg_latency']:.3f}s | {agg['avg_tokens']:.1f} | ${agg['total_cost']:.4f} | ${agg['cost_per_success']:.4f} | {agg.get('token_savings_pct', 0.0):.1f}% | {agg['local_percent']:.1f}% | {agg['remote_percent']:.1f}% | {agg['fallback_percent']:.1f}% |")
    report.append("\n")
    
    report.append("## Performance Visualizations\n")
    report.append("### Accuracy vs. Total API Cost\n")
    report.append('<div style="display: flex; gap: 20px; flex-wrap: wrap;">')
    report.append('  <img src="accuracy_chart.svg" width="48%" alt="Accuracy Chart" />')
    report.append('  <img src="cost_chart.svg" width="48%" alt="Cost Chart" />')
    report.append('</div>\n')
    
    # Oracle Routing Summary Section
    report.append("# Oracle Routing Summary\n")
    report.append(f"- **Local Optimal**: `{local_opts}` / `{executed_size}` ({local_opts/executed_size*100:.1f}%)")
    report.append(f"- **Remote Optimal**: `{remote_opts}` / `{executed_size}` ({remote_opts/executed_size*100:.1f}%)")
    report.append(f"- **No Successful Route**: `{no_opts}` / `{executed_size}` ({no_opts/executed_size*100:.1f}%)")
    report.append(f"- **Predictive Routing Accuracy**: `{pred_routing_accuracy:.1f}%` (calculated only on solvable/comparable prompts)")
    report.append(f"- **Over-Routing Ratio**: `{over_routing_ratio:.1f}%` (remote chosen but local or none was optimal)")
    report.append(f"- **Under-Routing Ratio**: `{under_routing_ratio:.1f}%` (local chosen but remote was optimal)")
    report.append(f"- **Routing Efficiency (Oracle / Predictive)**: `{routing_efficiency:.3f}`")
    report.append(f"- **Token Efficiency**: `{token_efficiency_pct:.1f}%` (Oracle tokens / Predictive tokens)")
    report.append(f"- **Oracle Cost**: `${oracle_cost:.4f}`")
    report.append(f"- **Predictive Cost**: `${pred_total_cost:.4f}`")
    report.append(f"- **Cost Ratio (Predictive / Oracle)**: `{cost_ratio:.2f}`")
    report.append(f"- **Estimated Savings**: `{pred_agg['cost_savings_pct']:.1f}%` (Cost savings compared to always-remote)")
    report.append("\n")
    
    # Per-Category Routing Distribution
    report.append("## Per-Category Routing Distribution (Predictive Strategy)\n")
    if category_metrics:
        report.append("| Category | Count | Accuracy | Avg Tokens | Avg Latency | Local % | Remote % | Repair % | Fallback % |")
        report.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for cat in sorted(category_metrics.keys()):
            m = category_metrics[cat]
            report.append(f"| {cat} | {m['count']} | {m['accuracy']:.1f}% | {m['avg_tokens']:.1f} | {m['avg_latency']:.3f}s | {m['local_pct']:.1f}% | {m['remote_pct']:.1f}% | {m['repair_pct']:.1f}% | {m['fallback_pct']:.1f}% |")
    else:
        report.append("*No per-category metrics available (predictive strategy was not executed).*\n")
    report.append("\n")
    
    report.append("## Difficulty Breakdown (Predictive Strategy)\n")
    if difficulty_metrics:
        report.append("| Difficulty | Count | Accuracy | Avg Tokens | Total Cost |")
        report.append("| --- | --- | --- | --- | --- |")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_metrics:
                m = difficulty_metrics[diff]
                report.append(f"| {diff.capitalize()} | {m['count']} | {m['accuracy']:.1f}% | {m['avg_tokens']:.1f} | ${m['cost']:.4f} |")
    else:
        report.append("*No difficulty metrics available (predictive strategy was not executed).*\n")
    report.append("\n")
    
    report.append("## Failure Analysis & Edge Cases\n")
    
    report.append("### Top 20 Routing Mistakes\n")
    if not routing_mistakes:
        report.append("*No routing mistakes identified in this run.*\n")
    else:
        report.append("| ID | Prompt | Decision | Cost-Optimal Route | Complexity | Confidence | Cost Wasted | Score |")
        report.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in routing_mistakes:
            wasted = r["cost_dollars"] if r["route_chosen_normalized"] == "remote" and r["cost_optimal_route"] == "local" else 0.0
            report.append(f"| `{r['id']}` | {r['prompt'][:80]}... | `{r['route_chosen_normalized']}` | `{r['cost_optimal_route']}` | {r['complexity_score']:.2f} | {r['routing_confidence']:.2f} | ${wasted:.4f} | {r['evaluator_score']:.2f} |")
        report.append("\n")
        
    report.append("### Top 20 Highest-Token Prompts\n")
    if not highest_token_prompts:
        report.append("*No data available.*\n")
    else:
        report.append("| ID | Prompt | Category | Difficulty | Chosen Route | Tokens Used | Cost |")
        report.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in highest_token_prompts:
            report.append(f"| `{r['id']}` | {r['prompt'][:80]}... | {r['category']} | {r['difficulty']} | `{r['route_chosen_normalized']}` | {r['total_tokens']} | ${r['cost_dollars']:.4f} |")
        report.append("\n")
    
    report.append("### Top 20 Unnecessary Remote Calls\n")
    if not unnecessary_remote:
        report.append("*No unnecessary remote calls identified in this run.*\n")
    else:
        report.append("| ID | Prompt | Category | Complexity | Confidence | Tokens Used | Cost |")
        report.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in unnecessary_remote:
            report.append(f"| `{r['id']}` | {r['prompt'][:80]}... | {r['category']} | {r['complexity_score']:.2f} | {r['routing_confidence']:.2f} | {r['total_tokens']} | ${r['cost_dollars']:.4f} |")
        report.append("\n")
        
    report.append("### Top 20 Local Failures\n")
    if not local_failures:
        report.append("*No local failures identified in this run.*\n")
    else:
        report.append("| ID | Prompt | Complexity | Risk | Score | Repair Used | Fallback Occurred | Errors |")
        report.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in local_failures:
            report.append(f"| `{r['id']}` | {r['prompt'][:80]}... | {r['complexity_score']:.2f} | {r['risk_score']:.2f} | {r['evaluator_score']:.2f} | {r['repair_used']} | {r['remote_fallback_occurred']} | {r['errors'][:40]} |")
        report.append("\n")
        
    report.append("### Top 20 Evaluator Mistakes (Discrepancies)\n")
    if not evaluator_mistakes:
        report.append("*No significant evaluator discrepancies found.*\n")
    else:
        report.append("| ID | Prompt | Score | Result | Suspicion Type | Response Snippet |")
        report.append("| --- | --- | --- | --- | --- | --- |")
        for r in evaluator_mistakes:
            resp_snip = (r.get("response") or "")[:50].replace("\n", " ")
            report.append(f"| `{r['id']}` | {r['prompt'][:80]}... | {r['evaluator_score']:.2f} | `{'PASS' if r['evaluator_passed'] else 'FAIL'}` | {r.get('suspicion_reason', 'Discrepancy')} | {resp_snip}... |")
        report.append("\n")
        
    report.append("## Threshold Tuning & Optimization Recommendations\n")
    
    # Threshold recommendation logic based on actual borderline cases
    failed_local_borderline = 0
    passed_local_borderline = 0
    if "predictive" in results_by_strategy:
        for r in results_by_strategy["predictive"]:
            if r["complexity_score"] >= 0.30 and r["complexity_score"] <= 0.55:
                if r["cost_optimal_route"] == "remote" and r["route_chosen_normalized"] == "local":
                    failed_local_borderline += 1
                elif r["cost_optimal_route"] == "local" and r["route_chosen_normalized"] == "remote":
                    passed_local_borderline += 1
                
    if failed_local_borderline > passed_local_borderline * 1.2:
        tuning_rec = (
            "We recommend **lowering the ZONE_REMOTE_LIMIT** (currently 0.55) or **lowering the RISK_THRESHOLD** (currently 0.35). "
            f"There are {failed_local_borderline} borderline cases that were routed locally but failed validation and required fallback. "
            "Lowering limits will push these risky cases to remote immediately."
        )
    elif passed_local_borderline > failed_local_borderline * 1.2:
        tuning_rec = (
            "We recommend **raising the ZONE_LOCAL_LIMIT** (currently 0.30) or **raising the RISK_THRESHOLD** (currently 0.35). "
            f"There are {passed_local_borderline} borderline cases that were routed to the remote model, but the local model was fully capable "
            "of passing. Raising limits will yield higher cost savings by routing these cases locally."
        )
    else:
        tuning_rec = (
            "The current thresholds are well-balanced. There is a symmetric distribution of boundary risk."
        )
        
    report.append(f"> [!IMPORTANT]\n> **Tuning Strategy**:\n> {tuning_rec}\n")
    report.append("### Recommendations to improve prompts / validation rules:\n")
    report.append("1. **Enhance Regex and Schema Validation**: For coding and JSON categories, small syntax errors can cause cascading validation failures. Strengthening repair templates to output only raw block values reduces local failure rates.")
    report.append("2. **Add Strict Token Limits**: Some creative writing tasks generate very large output sequences locally, raising latencies without improving accuracy. Enforce maximum completion tokens under `always_local` execution.")
    report.append("3. **Disable Local Self-Critique for Borderline Cases**: When a query is in the borderline zone, running local self-critique increases local token usage significantly. It is more cost-effective to either skip self-critique or route directly to remote.")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(S.good(f"\n[OK] Benchmark Report written to: {report_path}"))
    print(S.good(f"[OK] CSV results exported to: {csv_path}"))
    print(S.good(f"[OK] JSON responses exported to: {responses_path}"))
    
    # Console Summary Print
    print("\n" + "=" * 125)
    print(S.header("                                                  BENCHMARK SUMMARY"))
    print("=" * 125)
    print(f"Dataset Size : {total_dataset_size} prompts")
    print(f"Executed     : {executed_size}")
    print(f"Skipped      : {skipped_size}")
    if args.limit:
        print(S.warn("Mode         : Partial Benchmark (Fast Development Mode)"))
    else:
        print(S.good("Mode         : Full Benchmark Run"))
    print("-" * 125)
    print(f"{'Strategy':<12} | {'Accuracy':<8} | {'Avg Score':<9} | {'Avg Overhead':<12} | {'Avg Model Lat':<13} | {'Avg Eval Lat':<12} | {'Avg Total Lat':<13} | {'Avg Tokens':<10} | {'Total Cost':<10} | {'Cost/Success':<12}")
    print("-" * 125)
    for strat in strategies:
        agg = aggregates[strat]
        print(f"{strat.replace('always_', '').capitalize():<12} | {agg['accuracy']:>7.1f}% | {agg['avg_score']:>9.2f} | {agg['avg_overhead']:>10.3f}s | {agg['avg_model_latency']:>11.3f}s | {agg['avg_eval_latency']:>10.3f}s | {agg['avg_latency']:>11.3f}s | {agg['avg_tokens']:>10.1f} | ${agg['total_cost']:>9.4f} | ${agg['cost_per_success']:>10.4f}")
    print("-" * 125)
    print("Evaluation Statuses:")
    for strat in strategies:
        agg = aggregates[strat]
        print(f"  - {strat.replace('always_', '').capitalize():<12}: Passed: {agg['passed_count']}, Failed: {agg['failed_count']}, Eval Errors: {agg['eval_error_count']}, Model Errors: {agg['model_error_count']}")
    print("=" * 125)
    
    print(f"Predictive Routing Accuracy        : {S.good(f'{pred_routing_accuracy:.1f}%')}")
    print(f"Routing Efficiency                 : {S.good(f'{routing_efficiency:.3f}')}")
    print(f"Token Efficiency                   : {S.good(f'{token_efficiency_pct:.1f}%')}")
    print(f"Cost Ratio (Predictive / Oracle)   : {S.good(f'{cost_ratio:.2f}')}")
    print(f"Cost Efficiency                    : {S.good(f'{cost_efficiency_pct:.1f}%')}")
    print("=" * 125 + "\n")

    # -------------------------------------------------------------
    # Threshold Tuning & Calibration Logic
    # -------------------------------------------------------------
    if "predictive" in results_by_strategy and len(results_by_strategy["predictive"]) > 0:
        pred_recs = results_by_strategy["predictive"]
        
        # Retrieve always_local and always_remote responses
        always_local_recs = []
        always_remote_recs = []
        for i in range(executed_size):
            query = dataset[i]["prompt"]
            cat = dataset[i]["category"]
            fmt = get_response_format(cat, query)
            
            # local
            local_rec = None
            if "always_local" in results_by_strategy:
                local_rec = results_by_strategy["always_local"][i]
            else:
                cached = router.cache.get(query, "always_local", fmt)
                if cached:
                    local_rec = cached
            if not local_rec:
                local_rec = {"evaluator_passed": False, "total_latency_sec": 5.0, "cost_dollars": 0.0}
            always_local_recs.append(local_rec)
            
            # remote
            remote_rec = None
            if "always_remote" in results_by_strategy:
                remote_rec = results_by_strategy["always_remote"][i]
            else:
                cached = router.cache.get(query, "always_remote", fmt)
                if cached:
                    remote_rec = cached
            if not remote_rec:
                remote_rec = {"evaluator_passed": False, "total_latency_sec": 3.0, "cost_dollars": 0.001}
            always_remote_recs.append(remote_rec)

        # Determine baseline oracle routing accuracy
        baseline_comparable = [r for r in pred_recs if r["cost_optimal_route"] in ("local", "remote")]
        baseline_routing_matches = sum(1 for r in baseline_comparable if r["routing_match"])
        baseline_oracle_accuracy = (baseline_routing_matches / len(baseline_comparable)) if baseline_comparable else 0.0
             # Run optimization search
        step = args.calibration_step
        candidates = [round(x * step, 4) for x in range(int(1.0 / step) + 1)]
        
        # Bounded search space
        local_candidates = [c for c in candidates if 0.10 <= c <= 0.80]
        remote_candidates = [c for c in candidates if 0.15 <= c <= 0.90]
        risk_candidates = [c for c in candidates if 0.10 <= c <= 0.90]
        
        default_local = 0.30
        default_remote = 0.55
        default_risk = 0.35
        
        # Load from file if exists to compare correctly
        if os.path.exists("calibrated_thresholds.json"):
            try:
                with open("calibrated_thresholds.json", "r") as f_curr:
                    curr_d = json.load(f_curr)
                    def get_stored_threshold(key, default_val):
                        val = curr_d.get(key)
                        if val is not None:
                            return val
                        metadata = curr_d.get("metadata")
                        if isinstance(metadata, dict):
                            calib = metadata.get("calibrated_thresholds")
                            if isinstance(calib, dict) and calib.get(key) is not None:
                                return calib.get(key)
                        calib = curr_d.get("calibrated_thresholds")
                        if isinstance(calib, dict) and calib.get(key) is not None:
                            return calib.get(key)
                        return default_val
                    default_local = get_stored_threshold("ZONE_LOCAL_LIMIT", default_local)
                    default_remote = get_stored_threshold("ZONE_REMOTE_LIMIT", default_remote)
                    default_risk = get_stored_threshold("RISK_THRESHOLD", default_risk)
            except Exception:
                pass
                
        baseline_passed = sum(1 for r in pred_recs if r.get("evaluator_passed") is True)
        baseline_accuracy = baseline_passed / len(pred_recs) if pred_recs else 0.0
        baseline_cost = sum(r.get("cost_dollars", 0.0) for r in pred_recs)
        baseline_latency = sum(r.get("total_latency_sec", 0.0) for r in pred_recs)
        
        baseline_local_decisions = sum(1 for r in pred_recs if r.get("route_chosen_normalized") == "local")
        baseline_local_passed = sum(1 for r in pred_recs if r.get("route_chosen_normalized") == "local" and r.get("evaluator_passed") is True)
        baseline_local_success_rate = baseline_local_passed / baseline_local_decisions if baseline_local_decisions > 0 else 1.0
        baseline_local_pct = (baseline_local_decisions / len(pred_recs) * 100) if pred_recs else 0.0
        baseline_remote_pct = 100.0 - baseline_local_pct
        
        solvable_prompts = [idx for idx, r in enumerate(pred_recs) if cost_optimal_routes[r["id"]] in ("local", "remote")]
        total_solvable = len(solvable_prompts)
        
        best_local = default_local
        best_remote = default_remote
        best_risk = default_risk
        
        best_oracle_accuracy = baseline_oracle_accuracy
        best_oracle_matches = baseline_routing_matches
        best_cost = baseline_cost
        best_latency = baseline_latency
        best_local_pct = baseline_local_pct
        best_remote_pct = baseline_remote_pct
        best_accuracy = baseline_accuracy
        best_local_success_rate = baseline_local_success_rate
        
        # Pre-extract lists of evaluator success rates, latencies, and costs for always_local and always_remote
        local_passed_list = [bool(rec.get("evaluator_passed")) for rec in always_local_recs]
        local_cost_list = [float(rec.get("cost_dollars", 0.0)) for rec in always_local_recs]
        local_latency_list = [float(rec.get("total_latency_sec", 0.0)) for rec in always_local_recs]
        
        remote_passed_list = [bool(rec.get("evaluator_passed")) for rec in always_remote_recs]
        remote_cost_list = [float(rec.get("cost_dollars", 0.0)) for rec in always_remote_recs]
        remote_latency_list = [float(rec.get("total_latency_sec", 0.0)) for rec in always_remote_recs]
        
        complexity_scores = [float(r["complexity_score"]) for r in pred_recs]
        risk_scores = [float(r["risk_score"]) for r in pred_recs]
        ground_truths = [cost_optimal_routes[r["id"]] for r in pred_recs]
        
        for l_lim in local_candidates:
            for r_lim in remote_candidates:
                if r_lim - l_lim < step - 1e-6:
                    continue
                for r_thresh in risk_candidates:
                    sim_passed = 0
                    sim_cost = 0.0
                    sim_latency = 0.0
                    sim_local_decisions = 0
                    sim_local_passed = 0
                    sim_oracle_matches = 0
                    pruned = False
                    
                    for idx in range(executed_size):
                        c = complexity_scores[idx]
                        risk = risk_scores[idx]
                        gt = ground_truths[idx]
                        
                        # Simulated decision
                        if c < l_lim:
                            decision = "local"
                        elif c >= r_lim:
                            decision = "remote"
                        else:
                            decision = "remote" if risk >= r_thresh else "local"
                            
                        # Accumulate
                        if decision == "local":
                            sim_local_decisions += 1
                            if local_passed_list[idx]:
                                sim_passed += 1
                                sim_local_passed += 1
                            sim_cost += local_cost_list[idx]
                            sim_latency += local_latency_list[idx]
                        else:
                            if remote_passed_list[idx]:
                                sim_passed += 1
                            sim_cost += remote_cost_list[idx]
                            sim_latency += remote_latency_list[idx]
                            
                        if gt in ("local", "remote") and decision == gt:
                            sim_oracle_matches += 1
                            
                        # Early pruning checks
                        remaining = executed_size - 1 - idx
                        
                        # Constraint: Cost cannot increase
                        if sim_cost > baseline_cost + 1e-9:
                            pruned = True
                            break
                            
                        # Constraint: Latency increase must remain within tolerance
                        if sim_latency > baseline_latency * (1.0 + args.latency_tolerance):
                            pruned = True
                            break
                            
                        # Constraint: Overall accuracy (passed counts) cannot meet baseline
                        if (sim_passed + remaining) < baseline_passed:
                            pruned = True
                            break
                            
                        # Constraint: Oracle matches cannot reach the best found so far
                        if (sim_oracle_matches + remaining) < best_oracle_matches:
                            pruned = True
                            break
                            
                        # Constraint: Local success rate cannot meet baseline
                        sim_local_failures = sim_local_decisions - sim_local_passed
                        if sim_local_failures > 0:
                            max_local_success_rate = 1.0 - (sim_local_failures / (sim_local_decisions + remaining))
                            if max_local_success_rate < baseline_local_success_rate - 1e-6:
                                pruned = True
                                break
                                
                    if pruned:
                        continue
                        
                    sim_accuracy = sim_passed / executed_size if executed_size > 0 else 0.0
                    sim_oracle_accuracy = sim_oracle_matches / total_solvable if total_solvable > 0 else 0.0
                    sim_local_success_rate = sim_local_passed / sim_local_decisions if sim_local_decisions > 0 else 1.0
                    sim_local_pct = (sim_local_decisions / executed_size) * 100 if executed_size > 0 else 0.0
                    sim_remote_pct = 100.0 - sim_local_pct
                    
                    # Final constraints double check
                    if sim_accuracy < (baseline_accuracy - 1e-6):
                        continue
                    if sim_local_success_rate < (baseline_local_success_rate - 1e-6):
                        continue
                    if sim_cost > (baseline_cost + 1e-9):
                        continue
                    if sim_latency > (baseline_latency * (1.0 + args.latency_tolerance)):
                        continue
                        
                    is_better = False
                    if sim_oracle_accuracy > best_oracle_accuracy:
                        is_better = True
                    elif abs(sim_oracle_accuracy - best_oracle_accuracy) < 1e-5:
                        # Tie-breaker 1: Lower total cost
                        if sim_cost < best_cost:
                            is_better = True
                        elif abs(sim_cost - best_cost) < 1e-6:
                            # Tie-breaker 2: Lower latency
                            if sim_latency < best_latency:
                                is_better = True
                            elif abs(sim_latency - best_latency) < 1e-4:
                                # Tie-breaker 3: Greater local routing percentage
                                if sim_local_pct > best_local_pct:
                                    is_better = True
                                    
                    if is_better:
                        best_oracle_accuracy = sim_oracle_accuracy
                        best_oracle_matches = sim_oracle_matches
                        best_cost = sim_cost
                        best_latency = sim_latency
                        best_local_pct = sim_local_pct
                        best_remote_pct = sim_remote_pct
                        best_local = l_lim
                        best_remote = r_lim
                        best_risk = r_thresh
                        best_accuracy = sim_accuracy
                        best_local_success_rate = sim_local_success_rate

        print(S.header("                                                  CALIBRATION DIAGNOSTICS"))
        print("-" * 125)
        print(f"Current Thresholds  : ZONE_LOCAL_LIMIT={default_local:.2f}, ZONE_REMOTE_LIMIT={default_remote:.2f}, RISK_THRESHOLD={default_risk:.2f}")
        print(f"Optimal Thresholds  : ZONE_LOCAL_LIMIT={best_local:.2f}, ZONE_REMOTE_LIMIT={best_remote:.2f}, RISK_THRESHOLD={best_risk:.2f}")
        print(f"Oracle Routing Acc  : Baseline {baseline_oracle_accuracy * 100:.1f}% -> Calibrated {best_oracle_accuracy * 100:.1f}%")
        print("-" * 125)

        # Check safety persistence rules
        oracle_local_count = sum(1 for gt in cost_optimal_routes.values() if gt == "local")
        oracle_remote_count = sum(1 for gt in cost_optimal_routes.values() if gt == "remote")
        
        has_sufficient_prompts = (executed_size >= args.min_calibration_prompts)
        has_measurable_improvement = ((best_oracle_accuracy - baseline_oracle_accuracy) >= args.min_improvement)
        
        should_persist = (
            not args.dry_run_calibration and
            has_sufficient_prompts and
            has_measurable_improvement
        )
        
        if should_persist:
            calib_data = {
                "schema_version": 2,
                "metadata": {
                    "benchmark_version": args.benchmark_version,
                    "date": datetime.now().isoformat(),
                    "prompt_count": executed_size,
                    "optimization_score": float(best_oracle_accuracy),
                    "previous_thresholds": {
                        "ZONE_LOCAL_LIMIT": default_local,
                        "ZONE_REMOTE_LIMIT": default_remote,
                        "RISK_THRESHOLD": default_risk
                    },
                    "calibrated_thresholds": {
                        "ZONE_LOCAL_LIMIT": best_local,
                        "ZONE_REMOTE_LIMIT": best_remote,
                        "RISK_THRESHOLD": best_risk
                    },
                    "metrics_before_calibration": {
                        "oracle_routing_accuracy": float(baseline_oracle_accuracy),
                        "overall_accuracy": float(baseline_accuracy),
                        "total_cost": float(baseline_cost),
                        "total_latency": float(baseline_latency),
                        "local_success_rate": float(baseline_local_success_rate),
                        "local_routing_percentage": float(baseline_local_pct),
                        "remote_routing_percentage": float(baseline_remote_pct)
                    },
                    "metrics_after_calibration": {
                        "oracle_routing_accuracy": float(best_oracle_accuracy),
                        "overall_accuracy": float(best_accuracy),
                        "total_cost": float(best_cost),
                        "total_latency": float(best_latency),
                        "local_success_rate": float(best_local_success_rate),
                        "local_routing_percentage": float(best_local_pct),
                        "remote_routing_percentage": float(best_remote_pct)
                    }
                },
                "ZONE_LOCAL_LIMIT": best_local,
                "ZONE_REMOTE_LIMIT": best_remote,
                "RISK_THRESHOLD": best_risk
            }
            try:
                with open("calibrated_thresholds.json", "w", encoding="utf-8") as f_cal:
                    json.dump(calib_data, f_cal, indent=2)
                print(S.good("[SUCCESS] New thresholds persisted to calibrated_thresholds.json"))
            except Exception as e:
                print(S.warn(f"Failed to persist thresholds: {e}"))
        else:
            reasons = []
            if args.dry_run_calibration:
                reasons.append("Dry-run calibration flag set")
            if not has_sufficient_prompts:
                reasons.append(f"Prompt count ({executed_size}) < minimum required ({args.min_calibration_prompts})")
            if not has_measurable_improvement:
                reasons.append(f"Improvement ({(best_oracle_accuracy - baseline_oracle_accuracy)*100:.1f}%) < required margin ({args.min_improvement * 100:.1f}%)")
                
            print(S.warn(f"⚠️ Threshold calibration NOT persisted. Reasons: {', '.join(reasons)}"))
            
        # Compile running evaluator statistics
        eval_successes = {}
        eval_totals = {}
        
        for i in range(executed_size):
            rec = always_local_recs[i]
            query = dataset[i]["prompt"]
            cat = dataset[i]["category"]
            fmt = get_response_format(cat, query)
            selected_eval = get_selected_evaluator(query, fmt)
            
            passed = bool(rec.get("evaluator_passed"))
            eval_names = selected_eval.split("+")
            for name in eval_names:
                eval_successes[name] = eval_successes.get(name, 0) + (1 if passed else 0)
                eval_totals[name] = eval_totals.get(name, 0) + 1
                
        stats_file = "evaluation_stats.json"
        existing_stats = {}
        if os.path.exists(stats_file):
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    existing_stats = json.load(f)
            except Exception:
                pass
                
        for name in eval_totals:
            entry = existing_stats.get(name, {})
            if not isinstance(entry, dict):
                entry = {}
            success_count = entry.get("success_count", 0) + eval_successes[name]
            total_count = entry.get("total_count", 0) + eval_totals[name]
            success_rate = (success_count / total_count) if total_count > 0 else 0.0
            
            existing_stats[name] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": round(success_rate, 4)
            }
            
        if not args.dry_run_calibration:
            try:
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(existing_stats, f, indent=2)
                print(S.good(f"[SUCCESS] Updated running evaluator statistics in {stats_file}"))
            except Exception as e:
                print(S.warn(f"Failed to save evaluator statistics: {e}"))
                
        print("=" * 125 + "\n")



if __name__ == "__main__":
    run_benchmark()
