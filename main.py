import sys
import os
import argparse
import json
from src.client import LLMClient
from src.router import HybridRouter, RouterCache
from src.style import S

def parse_schema(schema_str: str):
    if not schema_str:
        return None
    try:
        return json.loads(schema_str)
    except json.JSONDecodeError:
        pass
    if ',' in schema_str or schema_str.isalnum():
        return [k.strip() for k in schema_str.split(',') if k.strip()]
    return schema_str

# Detect terminal width for dynamic separators
_TERM_WIDTH = 80
try:
    _TERM_WIDTH = os.get_terminal_size().columns
except (ValueError, OSError):
    _TERM_WIDTH = 80
SEP_WIDTH = max(min(_TERM_WIDTH, 100), 50)
BAR = S.bar("=" * SEP_WIDTH)
DASH = S.bar("-" * SEP_WIDTH)

def print_system_banner(client: LLMClient):
    """Print system info banner including GPU detection."""
    gpu = client.gpu_info
    pad = " " * 4

    print()
    print(f"{pad}{S.bold('System')}")
    print(f"{pad}  {S.label('GPU:')}       ", end="")
    if gpu["available"]:
        vram_str = f" ({gpu['vram_gb']} GB VRAM)" if gpu["vram_gb"] else ""
        print(S.good(gpu['name']) + S.muted(vram_str))
    else:
        print(S.muted("Not detected (running on CPU)"))
    print(f"{pad}  {S.label('Local:')}     {S.value(client.local_model)}")
    print(f"{pad}  {S.label('Remote:')}    {S.value(client.remote_model)}")
    if gpu["driver"]:
        print(f"{pad}  {S.label('ROCm:')}      {S.muted(gpu['driver'])}")


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

    if args.clear_cache:
        cache = RouterCache()
        cache.clear()
        print(S.good("Local persistent cache cleared."))
        sys.exit(0)

    parsed_schema = parse_schema(args.schema)
    client = LLMClient()

    if not args.query:
        print()
        print(BAR)
        print(S.header("  Hybrid Token-Efficient Routing Agent"))
        print(BAR)
        print_system_banner(client)
        print()
        print(f"  {S.label('Strategy:')}  {S.value(args.strategy)}  {S.muted('|')}  {S.label('Format:')}  {S.value(args.format)}")
        if parsed_schema:
            print(f"  {S.label('Schema:')}   {S.value(str(parsed_schema))}")
        print(f"  {S.label('Cache:')}    {S.value('ON' if not args.no_cache else 'OFF')}")
        print(f"  {S.muted('Type exit or quit to stop.')}")
        print(BAR)

        router = HybridRouter(client)

        while True:
            try:
                query = input(f"\n{S.accent('Query')} {S.accent('▸ ')}").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    break
                print(f"  {S.muted('Routing and processing...')}")
                res = router.route_and_execute(
                    query, strategy=args.strategy, response_format=args.format,
                    schema=parsed_schema, no_cache=args.no_cache
                )
                print_metrics(res)
            except KeyboardInterrupt:
                print(f"\n  {S.warn('Exiting...')}")
                break
            except Exception as e:
                print(f"\n  {S.error(f'Error: {e}')}")
    else:
        print()
        print(BAR)
        print(S.header("  Hybrid Token-Efficient Routing Agent"))
        print(BAR)
        router = HybridRouter(client)
        res = router.route_and_execute(
            args.query, strategy=args.strategy, response_format=args.format,
            schema=parsed_schema, no_cache=args.no_cache
        )
        print_metrics(res)


def _colour_route(route: str) -> str:
    """Return a colour-formatted route name."""
    if "local_best_effort" in route:
        return S.warn("local (best effort)")
    if "remote_fallback" in route:
        return S.warn("remote (fallback)")
    if "remote_failed" in route:
        return S.error("remote (failed)")
    if "remote" in route:
        return S.value("remote")
    if "local" in route:
        return S.good("local")
    return S.value(route)


def print_metrics(result: dict):
    p = "  "
    r = result

    print()
    print(BAR)
    print(S.header("  EXECUTION METRICS"))
    print(BAR)

    route = r['route_chosen'] or "unknown"
    cached = r.get('cached', False)
    route_display = _colour_route(route)
    if cached:
        route_display += S.muted(" (cached)")

    print(f"{p}{S.label('Route')}             {route_display}")
    print(f"{p}{S.label('Local attempts')}    {S.value(str(r['local_attempts']))}")
    print(f"{p}{S.label('Remote attempts')}   {S.value(str(r['remote_attempts']))}")
    fb = 'Yes' if r.get('fallback_triggered') else 'No'
    fb_s = S.warn(fb) if r.get('fallback_triggered') else S.muted(fb)
    print(f"{p}{S.label('Fallback triggered')} {fb_s}")
    c_s = S.good('Yes') if cached else S.muted('No')
    print(f"{p}{S.label('Cached')}            {c_s}")
    lat = r.get('latency_sec', 0.0)
    print(f"{p}{S.label('Latency')}           {S.value(f'{lat:.3f}s')}")

    local_tok = r['prompt_tokens_local'] + r['completion_tokens_local']
    remote_tok = r['prompt_tokens_remote'] + r['completion_tokens_remote']
    total_tok = local_tok + remote_tok

    local_detail = f'(prompt: {r["prompt_tokens_local"]}, completion: {r["completion_tokens_local"]})'
    remote_detail = f'(prompt: {r["prompt_tokens_remote"]}, completion: {r["completion_tokens_remote"]})'
    print(f"{p}{S.label('Local tokens')}      {S.value(str(local_tok))} {S.muted(local_detail)}")
    print(f"{p}{S.label('Remote tokens')}     {S.value(str(remote_tok))} {S.muted(remote_detail)}")
    print(f"{p}{S.label('Total tokens')}      {S.value(str(total_tok))}")

    cost = r.get('cost_dollars', 0.0)
    print(f"{p}{S.label('Remote cost')}       {S.money(f'${cost:.8f}')}")

    local_tok = r['prompt_tokens_local'] + r['completion_tokens_local']
    saving = r['cost_saved'] * 100
    if saving > 0:
        print(f"{p}{S.label('Token cost saved')}  {S.good(f'{saving:.1f}%')}")
        remote_price = float(os.getenv("REMOTE_PRICE_PER_1M_TOKENS", "0.90"))
        est = (local_tok / 1_000_000.0) * remote_price
        print(f"{p}{S.label('Estimated savings')} {S.money(f'${est:.8f}')} {S.muted('(vs. routing all tokens to remote)')}")
    else:
        print(f"{p}{S.label('Token cost saved')}  {S.value('0.0%')}")

    print(DASH)
    print(S.sub_header("  RESPONSE"))
    print(DASH)
    print(r['response'])
    print(BAR)
    print()


if __name__ == "__main__":
    main()
