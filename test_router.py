import os
import time
from src.evaluator import ResponseEvaluator
from src.client import LLMClient
from src.router import HybridRouter, RoutingDiagnostics, ComplexityReport

def test_routing_diagnostics():
    print("--- Running Routing Diagnostics Unit Tests ---")
    router = HybridRouter()
    query = "Write a python function to compute the factorial of a number using recursion."
    
    # Verify we can unpack it as a tuple
    report, diagnostics = router.analyze_complexity(query, response_format="python")
    
    assert report is not None
    assert diagnostics is not None
    
    # Check diagnostics fields
    assert diagnostics.query == query
    assert diagnostics.total_complexity_score == report.score
    assert isinstance(diagnostics.feature_scores, dict)
    assert diagnostics.selected_route in ("local", "remote")
    assert diagnostics.threshold_used > 0.0
    assert len(diagnostics.explanation) > 0
    
    # Check that individual feature contributes
    assert len(diagnostics.feature_scores) > 0
    for k, v in diagnostics.feature_scores.items():
        assert v >= 0.0
        
    print("[OK] Routing Diagnostics Unit Tests PASSED!")

def test_response_evaluator():
    evaluator = ResponseEvaluator()
    print("--- Running ResponseEvaluator Unit Tests ---")
    
    # 1. Repetition loop detection
    text_with_loop = "The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo."
    text_with_loop_2 = "Hello hello hello hello hello"
    is_valid, err = evaluator.evaluate("Where is the capital of Japan?", text_with_loop)
    is_valid_2, err_2 = evaluator.evaluate("Say hello", text_with_loop_2)
    assert not is_valid and "repetitive loops" in err.lower()
    assert not is_valid_2 and "repetitive loops" in err_2.lower()

    # 2. Summarization Sentence constraints
    query = "Summarize the text in one sentence."
    response_ok = "The quick brown fox jumps over the lazy dog."
    response_bad = "The quick brown fox jumps over the lazy dog. It was very fast."
    is_valid_ok, _ = evaluator.evaluate(query, response_ok)
    is_valid_bad, err = evaluator.evaluate(query, response_bad)
    assert is_valid_ok
    assert not is_valid_bad and "multiple sentences" in err.lower()

    # 3. Summarization Word limit constraints
    query_limit = "Summarize this in under 5 words."
    response_limit_ok = "This is a summary."
    response_limit_bad = "This is a summary that is too long."
    is_valid_lim_ok, _ = evaluator.evaluate(query_limit, response_limit_ok)
    is_valid_lim_bad, err = evaluator.evaluate(query_limit, response_limit_bad)
    assert is_valid_lim_ok
    assert not is_valid_lim_bad and "word limit" in err.lower()

    # 4. Summarization Bullet points list
    query_list = "List the features using bullet points."
    response_list_ok = "- Feature 1\n- Feature 2"
    response_list_bad = "Feature 1 and Feature 2 are good."
    is_valid_list_ok, _ = evaluator.evaluate(query_list, response_list_ok)
    is_valid_list_bad, err = evaluator.evaluate(query_list, response_list_bad)
    assert is_valid_list_ok
    assert not is_valid_list_bad and "bullet points" in err.lower()

    # 5. Sentiment justification
    query_sentiment = "Classify the sentiment and justify your choice."
    response_sent_ok = "The sentiment is positive because the user is extremely happy with the product."
    response_sent_bad = "Positive."
    is_valid_sent_ok, _ = evaluator.evaluate(query_sentiment, response_sent_ok)
    is_valid_sent_bad, err = evaluator.evaluate(query_sentiment, response_sent_bad)
    assert is_valid_sent_ok
    assert not is_valid_sent_bad and "justification" in err.lower()

    # 6. Quality scoring and rich fields check
    query_comp = "Compare Postgres and Redis in terms of latency, and recommend one."
    response_comp_good = (
        "Postgres has higher latency due to disk storage. "
        "Redis has lower latency since it is in-memory. "
        "I recommend Redis for latency-critical tasks."
    )
    result = evaluator.evaluate(query_comp, response_comp_good)
    assert hasattr(result, "quality_score")
    assert result.passed
    assert result.quality_score > 0.8
    assert result.confidence >= 0.85
    assert len(result.strengths) > 0
    # verify unpacking compatibility
    passed_unpacked, err_unpacked = result
    assert passed_unpacked
    assert err_unpacked == ""

    # Test missing dimension / entity quality decrease
    response_comp_bad = "Postgres is a database. I recommend Postgres."
    result_bad = evaluator.evaluate(query_comp, response_comp_bad)
    assert result_bad.quality_score < result.quality_score
    assert len(result_bad.failure_reasons) > 0

    # 7. Configurable weights
    custom_weights = {
        "Formatting": 0.1,
        "Task Completion": 0.1,
        "Coverage": 0.1,
        "Reasoning": 0.1,
        "Constraint Satisfaction": 0.1,
        "Output Structure": 0.1,
        "Completeness": 5.0, # High weight on Completeness
    }
    evaluator_custom = ResponseEvaluator(weights=custom_weights)
    result_custom = evaluator_custom.evaluate(query_comp, response_comp_good)
    assert abs(result_custom.quality_score - result_custom.component_scores["Completeness"]) < 0.1

    # 8. Placeholders rejection in Code Gen
    query_code = "Write a python function reverse_list to reverse a list."
    response_code_placeholder = (
        "def reverse_list(lst):\n"
        "    # TODO: implement\n"
        "    pass"
    )
    result_code = evaluator.evaluate(query_code, response_code_placeholder, response_format="python")
    assert not result_code.passed
    assert "placeholder" in result_code.critical_failures[0].lower()

    # 9. Nested JSON schema validation
    nested_schema = {
        "user": {
            "name": str,
            "id": int
        },
        "active": bool
    }
    response_json_ok = '{"user": {"name": "Alice", "id": 42}, "active": true}'
    response_json_bad = '{"user": {"name": "Alice", "id": "not_an_int"}, "active": true}'
    
    res_json_ok = evaluator.evaluate("Get user", response_json_ok, response_format="json", schema=nested_schema)
    res_json_bad = evaluator.evaluate("Get user", response_json_bad, response_format="json", schema=nested_schema)
    assert res_json_ok.passed
    assert not res_json_bad.passed
    assert "expected int" in res_json_bad.critical_failures[0].lower()

    # 10. Python AST function verification
    query_py = "Write a python function compute_sum."
    response_py_ok = "def compute_sum(a, b):\n    return a + b"
    response_py_bad = "def compute_diff(a, b):\n    return a - b"
    res_py_ok = evaluator.evaluate(query_py, response_py_ok, response_format="python")
    res_py_bad = evaluator.evaluate(query_py, response_py_bad, response_format="python")
    assert res_py_ok.passed
    assert not res_py_bad.passed
    assert any("missing in ast" in fr.lower() for fr in res_py_bad.failure_reasons)

    # 11. Architecture coverage-based component validation
    query_arch = "Design a system architecture with a database and monitoring."
    response_arch_db_only = "We use Postgres as our main database."
    res_arch = evaluator.evaluate(query_arch, response_arch_db_only)
    # Check that database is verified, monitoring is flagged as missing, but caching is NOT flagged (since it wasn't requested)
    assert any("monitoring" in fr.lower() for fr in res_arch.failure_reasons)
    assert not any("cache" in fr.lower() for fr in res_arch.failure_reasons)

    print("[OK] ResponseEvaluator Unit Tests PASSED!")

def run_predictive_scorer_benchmark():
    from src.router import HybridRouter
    router = HybridRouter()
    
    print("\n" + "="*60)
    print("RUNNING PREDICTIVE SCORER BENCHMARK SUITE")
    print("="*60)
    
    benchmark_cases = [
        {
            "category": "Factual Knowledge",
            "query": "What is Kafka?",
            "expected_decision": "local",
            "expected_zone": "Definitely Local"
        },
        {
            "category": "Mathematical Reasoning",
            "query": "Calculate: (3.14159 * 12.5^2) / 4",
            "expected_decision": "local",
            "expected_zone": "Borderline"
        },
        {
            "category": "Sentiment Classification",
            "query": "Classify sentiment of: 'Product is good.'",
            "expected_decision": "local",
            "expected_zone": "Definitely Local"
        },
        {
            "category": "Text Summarization",
            "query": "Summarize this 3000-word research paper into 10 bullet points including methodology, limitations and future work.",
            "expected_decision": "local",
            "expected_zone": "Borderline"
        },
        {
            "category": "Named Entity Recognition",
            "query": "Extract all cities from this sentence.",
            "expected_decision": "local",
            "expected_zone": "Definitely Local"
        },
        {
            "category": "Code Debugging",
            "query": "Fix memory leak in this C++ code block: [...]",
            "expected_decision": "remote",
            "expected_zone": "Definitely Remote"
        },
        {
            "category": "Logical Reasoning",
            "query": "If A is true, B is false, what is A and B?",
            "expected_decision": "local",
            "expected_zone": "Definitely Local"
        },
        {
            "category": "Code Generation",
            "query": "Write a Java function to reverse a string.",
            "expected_decision": "local",
            "expected_zone": "Definitely Local"
        }
    ]
    
    total_benchmark = len(benchmark_cases)
    passed_benchmark = 0
    total_time_ms = 0.0
    
    for idx, bc in enumerate(benchmark_cases, 1):
        print(f"\n[{idx}/{total_benchmark}] Benchmark: {bc['category']}")
        print(f"Query: '{bc['query']}'")
        
        # Measure latency
        start = time.perf_counter()
        report = router.analyze_complexity(bc['query'])
        latency_ms = (time.perf_counter() - start) * 1000.0
        total_time_ms += latency_ms
        
        print(f"-> Decision: {report.decision} (Expected: {bc['expected_decision']})")
        print(f"-> Zone: {report.routing_zone} (Expected: {bc['expected_zone']})")
        print(f"-> Score: {report.score:.2f} | Normalized: {report.normalized_score:.2f} | Risk Score: {report.risk_score:.2f}")
        print(f"-> Domains: {report.matched_domains} | Features: {report.matched_features}")
        print(f"-> Latency: {latency_ms:.3f} ms")
        
        if report.decision == bc['expected_decision']:
            print("Status: PASSED")
            passed_benchmark += 1
        else:
            print("Status: FAILED")
            
        # Assert execution speed is under 2 ms
        assert latency_ms < 2.0, f"Latency of {latency_ms:.2f} ms exceeds the 2 ms requirement!"
        
    avg_time = total_time_ms / total_benchmark
    print("\n" + "="*60)
    print("BENCHMARK SUITE SUMMARY")
    print("="*60)
    print(f"Passed: {passed_benchmark} / {total_benchmark}")
    print(f"Average Execution Speed: {avg_time:.3f} ms (Target: < 2.000 ms)")
    print("="*60 + "\n")
    
    assert passed_benchmark == total_benchmark, "Not all predictive routing decisions matched expectations!"

# Sample test cases representing different complexity levels, formats, and schemas
TEST_CASES = [
    {
        "name": "Simple Greeting",
        "query": "Hello! Who are you?",
        "format": "text",
        "expected_difficulty": "simple"
    },
    {
        "name": "Factual Question",
        "query": "What is the capital of Japan?",
        "format": "text",
        "expected_difficulty": "simple"
    },
    {
        "name": "JSON Output Generation",
        "query": "Create a JSON object listing 3 fruit names and their color. Format as JSON only, no markdown.",
        "format": "json",
        "expected_difficulty": "simple"
    },
    {
        "name": "JSON Schema Validation",
        "query": "Generate a JSON containing the fields 'fruit_name' and 'calories' representing a banana. Format as JSON only, no markdown.",
        "format": "json",
        "schema": {"fruit_name": str, "calories": int},
        "expected_difficulty": "simple"
    },
    {
        "name": "Python Code Syntax check",
        "query": "Write a python function to compute the factorial of a number using recursion.",
        "format": "python",
        "expected_difficulty": "complex"
    },
    {
        "name": "Complex Reasoning",
        "query": "A farmer has 17 sheep, and all but 9 run away. How many sheep are left? Explain your step-by-step logic carefully.",
        "format": "text",
        "expected_difficulty": "complex"
    }
]

def run_suite():
    print("="*60)
    print("RUNNING HYBRID ROUTER TEST SUITE (ENHANCED)")
    print("="*60)
    
    # Run unit tests
    test_routing_diagnostics()
    test_response_evaluator()
    
    # Run predictive scorer benchmark
    run_predictive_scorer_benchmark()
    
    # Check if Ollama is running and model is pulled
    client = LLMClient()
    router = HybridRouter(client)
    
    # Clear cache for deterministic testing
    print("Clearing persistent local cache for clean test run...")
    router.cache.clear()
    
    # Test checking
    try:
        print("Checking local Ollama connection...")
        client.call_local("ping", max_tokens=2)
        print("Local Ollama connection: OK")
    except Exception as e:
        print(f"Error: Local Ollama connection failed. Please make sure Ollama is running. Detail: {e}")
        return

    # Check Fireworks configuration
    has_remote = os.getenv("FIREWORKS_API_KEY") is not None and os.getenv("FIREWORKS_API_KEY") != ""
    if not has_remote:
        print("Warning: FIREWORKS_API_KEY not found in environment. Remote calls will be simulated/mocked for this test.")

    # We will test the 'fallback' strategy (which tries local first, validates, and falls back to remote)
    print("\nRunning test cases using 'fallback' strategy...")
    print("-" * 60)
    
    total_queries = len(TEST_CASES)
    local_successes = 0
    fallbacks = 0
    
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total_queries}] Testing: {tc['name']}")
        print(f"Query: '{tc['query']}'")
        schema_desc = f" | Schema: {tc['schema']}" if 'schema' in tc else ""
        print(f"Expected format: {tc['format']}{schema_desc} | Difficulty: {tc['expected_difficulty']}")
        
        # Override call_remote if key is missing to avoid crashing test suite
        if not has_remote:
            original_call_remote = client.call_remote
            client.call_remote = lambda q, *args, **kwargs: (
                f"[SIMULATED REMOTE RESPONSE for: {q[:30]}...]", 
                150, 
                50
            )

        # Run routing execution
        res = router.route_and_execute(
            tc['query'], 
            strategy="fallback", 
            response_format=tc['format'],
            schema=tc.get("schema")
        )
        
        print(f"-> Route Chosen: {res['route_chosen']}")
        print(f"-> Fallback Triggered: {res['fallback_triggered']}")
        print(f"-> Latency: {res['latency_sec']:.3f}s | Remote Cost: ${res['cost_dollars']:.6f}")
        
        if res['route_chosen'] in ["local", "local_fallback"]:
            local_successes += 1
            print("-> Result: Local model response passed validation!")
        else:
            fallbacks += 1
            print(f"-> Result: Routed to remote. Remote Tokens: {res['prompt_tokens_remote'] + res['completion_tokens_remote']}")

        # Verify caching mechanism
        print("-> Verifying cache hit on duplicate query...")
        res_cached = router.route_and_execute(
            tc['query'], 
            strategy="fallback", 
            response_format=tc['format'],
            schema=tc.get("schema")
        )
        print(f"   Duplicate Route: {res_cached['route_chosen']} | Cached: {res_cached['cached']} | Latency: {res_cached['latency_sec']:.6f}s")
        if not res_cached['cached']:
            print("   WARNING: Duplicate query was not served from cache!")

        # Restore original remote client call if mocked
        if not has_remote:
            client.call_remote = original_call_remote

        # Hardware cooldown sleep to protect low-spec machines from resource exhaustion
        time.sleep(0.5)
            
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Local Successes (0 cost): {local_successes} / {total_queries}")
    print(f"Remote Fallbacks:         {fallbacks} / {total_queries}")
    
    savings = (local_successes / total_queries) * 100
    print(f"Estimated Token Cost Reduction: {savings:.1f}%")
    print("="*60)

if __name__ == "__main__":
    run_suite()
