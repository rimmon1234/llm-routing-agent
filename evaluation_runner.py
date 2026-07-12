import os
import sys
import json
import traceback

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_evaluation():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"

    # Support relative paths for local development/testing convenience
    if not os.path.exists(input_path) and os.path.exists("./input/tasks.json"):
        input_path = "./input/tasks.json"
        output_path = "./output/results.json"

    # Fallback to interactive CLI if input file is not present (local dev mode only, not inside container)
    if not os.path.exists(input_path):
        if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1" or not sys.stdin.isatty():
            print(f"❌ Error: Input file {input_path} not found. Exiting with code 1.")
            sys.exit(1)
        print(f"ℹ️ Input file {input_path} not found. Falling back to interactive CLI main.py...")
        import subprocess
        # Pass CLI args along if any
        cmd = [sys.executable, "main.py"] + sys.argv[1:]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    print(f"📖 Reading tasks from {input_path}...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"❌ Error: Failed to read or parse {input_path}: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("❌ Error: tasks.json must contain a JSON list of tasks.")
        sys.exit(1)

    print(f"⚙️ Initializing LLM Client and Router...")
    try:
        from src.client import LLMClient
        from src.router import HybridRouter

        client = LLMClient()
        router = HybridRouter(client=client)
    except Exception as e:
        print(f"❌ Error: Failed to initialize routing framework: {e}")
        traceback.print_exc()
        sys.exit(1)

    results = []
    total_tasks = len(tasks)
    print(f"🚀 Processing {total_tasks} tasks...")

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        if not task_id or not prompt:
            print(f"⚠️ Warning [Task {idx}/{total_tasks}]: Missing task_id or prompt. Skipping.")
            continue

        print(f"   [{idx}/{total_tasks}] Routing Task ID: {task_id}")
        try:
            # We use 'predictive' strategy as our default evaluation strategy
            # Output format is text by default, let's let the router handle it
            res = router.route_and_execute(
                query=prompt,
                strategy="predictive",
                response_format="text"
            )
            answer = res.get("response", "")
            print(f"   Task {task_id}")
            print(f"   Generation: {res.get('generation_latency_sec', 0.0):.1f}s")
            print(f"   Evaluation: {res.get('evaluation_latency_sec', 0.0):.2f}s")
            print(f"   Total: {res.get('latency_sec', 0.0):.1f}s")
            print()
        except Exception as e:
            print(f"❌ Error during routing Task {task_id}: {e}")
            answer = f"Error generating answer: {str(e)}"

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Saving results to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✅ Evaluation complete. Exiting successfully (Code 0).")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: Failed to write results to {output_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
