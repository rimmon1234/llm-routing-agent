import os
import time
from src.client import LLMClient
from src.router import HybridRouter

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
