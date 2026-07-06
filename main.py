import sys
import argparse
import json
from src.client import LLMClient
from src.router import HybridRouter, RouterCache

def parse_schema(schema_str: str):
    if not schema_str:
        return None
    # Try parsing as JSON (dict/list) first
    try:
        return json.loads(schema_str)
    except json.JSONDecodeError:
        pass
    # Otherwise, split by comma if list of keys
    if ',' in schema_str or schema_str.isalnum():
        return [k.strip() for k in schema_str.split(',') if k.strip()]
    return schema_str

def main():
    parser = argparse.ArgumentParser(description="Hybrid Token-Efficient Routing Agent CLI")
    parser.add_argument("--query", type=str, help="The query/prompt to run")
    parser.add_argument("--strategy", type=str, default="fallback", 
                        choices=["always_local", "always_remote", "fallback", "predictive"],
                        help="Routing strategy (default: fallback)")
    parser.add_argument("--format", type=str, default="text",
                        choices=["text", "json", "python"],
                        help="Expected format of output for local validation (default: text)")
    parser.add_argument("--schema", type=str,
                        help="Required keys (comma-separated, e.g. 'name,age') or custom JSON schema for json validation")
    parser.add_argument("--no-cache", action="store_true", help="Bypass routing cache completely")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the persistent cache and exit")
    args = parser.parse_args()

    # Clear cache action
    if args.clear_cache:
        cache = RouterCache()
        cache.clear()
        print("Local persistent cache cleared.")
        sys.exit(0)

    parsed_schema = parse_schema(args.schema)

    # Interactive mode if no query is passed
    if not args.query:
        print("="*60)
        print("  Hybrid Token-Efficient Routing Agent CLI  ")
        print("="*60)
        print(f"Strategy: {args.strategy} | Expected Format: {args.format}")
        if parsed_schema:
            print(f"Validation Schema: {parsed_schema}")
        print(f"Cache: {'Disabled' if args.no_cache else 'Enabled'}")
        print("Type 'exit' or 'quit' to stop.\n")
        
        client = LLMClient()
        router = HybridRouter(client)
        
        while True:
            try:
                query = input("\nQuery > ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    break
                
                print("\nRouting and processing...")
                res = router.route_and_execute(
                    query, 
                    strategy=args.strategy, 
                    response_format=args.format,
                    schema=parsed_schema,
                    no_cache=args.no_cache
                )
                print_metrics(res)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
    else:
        client = LLMClient()
        router = HybridRouter(client)
        res = router.route_and_execute(
            args.query, 
            strategy=args.strategy, 
            response_format=args.format,
            schema=parsed_schema,
            no_cache=args.no_cache
        )
        print_metrics(res)

def print_metrics(result: dict):
    print("\n" + "="*50)
    print("EXECUTION METRICS")
    print("="*50)
    print(f"Route Chosen:       {result['route_chosen']}")
    print(f"Local Attempts:     {result['local_attempts']}")
    print(f"Remote Attempts:    {result['remote_attempts']}")
    print(f"Fallback Triggered: {result['fallback_triggered']}")
    print(f"Cached:             {result.get('cached', False)}")
    print(f"Latency:            {result.get('latency_sec', 0.0):.3f}s")
    
    local_tokens = result['prompt_tokens_local'] + result['completion_tokens_local']
    remote_tokens = result['prompt_tokens_remote'] + result['completion_tokens_remote']
    total_tokens = local_tokens + remote_tokens
    
    print(f"Local Tokens:       {local_tokens} (Prompt: {result['prompt_tokens_local']}, Completion: {result['completion_tokens_local']})")
    print(f"Remote Tokens:      {remote_tokens} (Prompt: {result['prompt_tokens_remote']}, Completion: {result['completion_tokens_remote']})")
    print(f"Total Tokens:       {total_tokens}")
    print(f"Remote Cost:        ${result.get('cost_dollars', 0.0):.6f}")
    
    saving = result['cost_saved'] * 100
    print(f"Token Cost Saved:   {saving:.1f}%")
    if result.get('estimated_savings_dollars', 0.0) > 0:
        print(f"Estimated Savings:  ${result['estimated_savings_dollars']:.6f} (vs. routing all tokens to remote)")
    print("-" * 50)
    print("RESPONSE:")
    print("-" * 50)
    print(result['response'])
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
