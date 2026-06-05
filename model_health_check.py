"""
═══════════════════════════════════════════════════════════════
  MODEL HEALTH CHECK — OpenRouter | Groq | Cohere
  Tests every model in the .env file and writes a status report
═══════════════════════════════════════════════════════════════
  Usage:
    1. Set your API keys below (or set them as environment variables)
    2. Make sure your .env file path is correct (ENV_FILE below)
    3. Run:  python model_health_check.py
    4. Report is saved as: model_health_report.txt
"""

import os
import re
import time
import json
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
#  API KEYS — replace the placeholders or set as env vars
# ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY",       "YOUR_GROQ_API_KEY_HERE")
COHERE_API_KEY     = os.getenv("COHERE_API_KEY",      "YOUR_COHERE_API_KEY_HERE")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
ENV_FILE        = ".env"   # path to your .env file
REPORT_FILE     = "model_health_report.txt"
REQUEST_TIMEOUT = 30      # seconds per request
MAX_WORKERS     = 6       # parallel threads (keep low to avoid rate limits)
RETRY_ATTEMPTS  = 2       # retries on timeout/network error
RETRY_DELAY     = 3       # seconds between retries

# Minimal test prompt — cheap, fast, and enough to confirm a model is alive
CHAT_TEST_PROMPT = "Reply with exactly: OK"

# ─────────────────────────────────────────────────────────────
#  PLATFORM DETECTION
# ─────────────────────────────────────────────────────────────
# These model IDs map to special endpoints (not standard chat)
COHERE_EMBED_MODELS = {
    # No Cohere embed models in the .env, but kept for extensibility
}

COHERE_RERANK_MODELS = {
    "cohere/rerank-4-pro",
    "cohere/rerank-4-fast",
    "cohere/rerank-v3.5",
}

GROQ_TTS_MODELS = {
    "canopylabs/orpheus-v1-english",
}

# Groq compound systems need a real search-triggering prompt (not a generic chat prompt)
# otherwise compound-mini times out waiting to invoke a tool
GROQ_COMPOUND_MODELS = {
    "groq/compound",
    "groq/compound-mini",
}

# OpenRouter embedding models (use /embeddings endpoint)
OPENROUTER_EMBED_MODELS = {
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "perplexity/pplx-embed-v1-0.6b",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "sentence-transformers/paraphrase-minilm-l6-v2",
    "sentence-transformers/all-minilm-l12-v2",
    "baai/bge-base-en-v1.5",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-minilm-l6-v2",
    "thenlper/gte-large",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large",
    "baai/bge-large-en-v1.5",
    "baai/bge-m3",
    "qwen/qwen3-embedding-8b",
    "openai/text-embedding-3-small",
    "qwen/qwen3-embedding-4b",
    "perplexity/pplx-embed-v1-4b",
}

# OpenRouter rerank models (use /rerank endpoint)
OPENROUTER_RERANK_MODELS = {
    "cohere/rerank-4-pro",
    "cohere/rerank-4-fast",
    "cohere/rerank-v3.5",
}


# ─────────────────────────────────────────────────────────────
#  STEP 1 — PARSE .ENV FILE
# ─────────────────────────────────────────────────────────────
def parse_env_file(filepath):
    """
    Reads the .env file and extracts all model entries.
    Returns a dict: { model_id: { 'platform': ..., 'categories': [...] } }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find .env file at: {filepath}")

    models = {}   # model_id -> { platform, categories: [list of env keys] }

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            # Match KEY=VALUE lines
            match = re.match(r'^([A-Z0-9_]+)\s*=\s*(.+)$', line)
            if not match:
                continue

            env_key   = match.group(1).strip()
            model_id  = match.group(2).strip()

            # Determine platform from key prefix
            if env_key.startswith("GROQ__"):
                platform = "groq"
            elif env_key.startswith("COHERE__"):
                platform = "cohere"
            else:
                # Default: OpenRouter (TEXT__, EMBEDDING__, RERANK__ prefixes)
                platform = "openrouter"

            if model_id not in models:
                models[model_id] = {"platform": platform, "categories": []}
            models[model_id]["categories"].append(env_key)

    return models


# ─────────────────────────────────────────────────────────────
#  STEP 2 — TEST FUNCTIONS PER PLATFORM + MODEL TYPE
# ─────────────────────────────────────────────────────────────

def _post_with_retry(url, headers, payload, timeout, retries, delay):
    """Shared HTTP POST with retry logic."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            return resp
        except requests.exceptions.Timeout:
            last_error = "TIMEOUT"
        except requests.exceptions.ConnectionError as e:
            last_error = f"CONNECTION_ERROR: {e}"
        if attempt < retries:
            time.sleep(delay)
    return last_error  # return string on total failure


def test_openrouter_chat(model_id):
    """Test a standard chat model via OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://model-health-check",
        "X-Title": "ModelHealthCheck",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": CHAT_TEST_PROMPT}],
        "max_tokens": 10,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    return _parse_chat_response(result, model_id)


def test_openrouter_embedding(model_id):
    """Test an embedding model via OpenRouter."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "input": "health check",
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            data = result.json()
            if "data" in data and len(data["data"]) > 0:
                return ("UP", f"HTTP 200 — embedding dim: {len(data['data'][0].get('embedding', []))}")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


def test_openrouter_rerank(model_id):
    """Test a rerank model via OpenRouter."""
    url = "https://openrouter.ai/api/v1/rerank"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "query": "health check",
        "documents": ["document one", "document two"],
        "top_n": 1,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            data = result.json()
            if "results" in data:
                return ("UP", f"HTTP 200 — rerank results: {len(data['results'])}")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


def test_groq_chat(model_id):
    """Test a Groq chat model."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": CHAT_TEST_PROMPT}],
        "max_tokens": 10,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    return _parse_chat_response(result, model_id)


def test_groq_compound(model_id):
    """Test a Groq compound system with a real search-triggering prompt.
    Plain chat prompts like 'Reply with OK' never invoke tools so compound-mini
    times out. A factual current-events question reliably triggers web search.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is today's date?"}],
        "max_tokens": 50,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT + 30, RETRY_ATTEMPTS, RETRY_DELAY)
    return _parse_chat_response(result, model_id)


def test_groq_tts(model_id):
    """Test a Groq TTS model (Orpheus) with a minimal synthesis call."""
    url = "https://api.groq.com/openai/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "input": "test",
        "voice": "autumn",      # valid voices: autumn, diana, hannah, austin, daniel, troy
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            content_len = len(result.content)
            return ("UP", f"HTTP 200 — audio bytes received: {content_len}")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


def test_cohere_chat(model_id):
    """Test a Cohere chat/command model."""
    url = "https://api.cohere.com/v2/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": CHAT_TEST_PROMPT}],
        "max_tokens": 10,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            data = result.json()
            # v2 response format
            content = ""
            if "message" in data:
                for block in data["message"].get("content", []):
                    if block.get("type") == "text":
                        content = block.get("text", "")[:40]
            return ("UP", f"HTTP 200 — response: '{content}'")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


def test_cohere_rerank(model_id):
    """Test a Cohere rerank model."""
    url = "https://api.cohere.com/v2/rerank"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "query": "health check",
        "documents": ["document one about health", "document two about systems"],
        "top_n": 1,
    }
    result = _post_with_retry(url, headers, payload, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY)
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            data = result.json()
            results = data.get("results", [])
            return ("UP", f"HTTP 200 — rerank results: {len(results)}")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _parse_chat_response(result, model_id):
    if isinstance(result, str):
        return ("DOWN", result)
    try:
        if result.status_code == 200:
            data = result.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                content = (content or "")[:40].replace("\n", " ")
                return ("UP", f"HTTP 200 — response: '{content}'")
            return ("UP", "HTTP 200 — no choices in response (may still be valid)")
        return ("DOWN", f"HTTP {result.status_code} — {_extract_error(result)}")
    except Exception as e:
        return ("DOWN", f"Parse error: {e}")


def _extract_error(response):
    try:
        data = response.json()
        # Try common error paths
        err = (
            data.get("error", {}).get("message")
            or data.get("message")
            or data.get("detail")
            or str(data)[:120]
        )
        return str(err)[:120]
    except Exception:
        return response.text[:120]


# ─────────────────────────────────────────────────────────────
#  STEP 3 — DISPATCH ROUTER
# ─────────────────────────────────────────────────────────────
def test_model(model_id, platform):
    """Route a model to the correct test function based on platform + type."""
    start = time.time()

    try:
        if platform == "openrouter":
            if model_id in OPENROUTER_EMBED_MODELS:
                model_type = "embedding"
                status, detail = test_openrouter_embedding(model_id)
            elif model_id in OPENROUTER_RERANK_MODELS:
                model_type = "rerank"
                status, detail = test_openrouter_rerank(model_id)
            else:
                model_type = "chat"
                status, detail = test_openrouter_chat(model_id)

        elif platform == "groq":
            if model_id in GROQ_TTS_MODELS:
                model_type = "tts"
                status, detail = test_groq_tts(model_id)
            elif model_id in GROQ_COMPOUND_MODELS:
                model_type = "compound"
                status, detail = test_groq_compound(model_id)
            else:
                model_type = "chat"
                status, detail = test_groq_chat(model_id)

        elif platform == "cohere":
            if model_id in COHERE_RERANK_MODELS:
                model_type = "rerank"
                status, detail = test_cohere_rerank(model_id)
            else:
                model_type = "chat"
                status, detail = test_cohere_chat(model_id)

        else:
            model_type = "unknown"
            status, detail = "SKIP", "Unknown platform"

    except Exception as e:
        model_type = "chat"
        status, detail = "DOWN", f"Unhandled exception: {e}"

    elapsed = round(time.time() - start, 2)
    return {
        "model_id": model_id,
        "platform": platform,
        "model_type": model_type,
        "status": status,
        "detail": detail,
        "latency_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────
#  STEP 4 — RUN ALL TESTS IN PARALLEL
# ─────────────────────────────────────────────────────────────
def run_all_tests(models_dict):
    """Run tests for all unique model_id+platform combos in parallel."""
    tasks = list(models_dict.items())  # [(model_id, {platform, categories})]
    results = []

    print(f"\n  Testing {len(tasks)} unique models across all platforms...")
    print(f"  Parallel workers: {MAX_WORKERS} | Timeout per request: {REQUEST_TIMEOUT}s\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_model = {
            executor.submit(test_model, model_id, info["platform"]): (model_id, info)
            for model_id, info in tasks
        }

        completed = 0
        for future in as_completed(future_to_model):
            model_id, info = future_to_model[future]
            completed += 1
            try:
                result = future.result()
                result["categories"] = info["categories"]
                results.append(result)
                icon = "✓" if result["status"] == "UP" else "✗"
                print(f"  [{completed:3}/{len(tasks)}] {icon} [{result['platform'].upper():12}] "
                      f"{result['status']:4} | {model_id} ({result['latency_s']}s)")
            except Exception as e:
                results.append({
                    "model_id": model_id,
                    "platform": info["platform"],
                    "model_type": "unknown",
                    "status": "DOWN",
                    "detail": f"Future error: {e}",
                    "latency_s": 0,
                    "categories": info["categories"],
                })
                print(f"  [{completed:3}/{len(tasks)}] ✗ [{info['platform'].upper():12}] "
                      f"DOWN | {model_id}")

    return results


# ─────────────────────────────────────────────────────────────
#  STEP 5 — BUILD AND WRITE REPORT
# ─────────────────────────────────────────────────────────────
def write_report(results, output_path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    up    = sum(1 for r in results if r["status"] == "UP")
    down  = sum(1 for r in results if r["status"] == "DOWN")
    skip  = sum(1 for r in results if r["status"] == "SKIP")

    # Group by platform
    platforms = ["openrouter", "groq", "cohere"]
    by_platform = {p: [r for r in results if r["platform"] == p] for p in platforms}

    lines = []
    def w(*args):
        lines.append(" ".join(str(a) for a in args))

    w("=" * 80)
    w("  MODEL HEALTH CHECK REPORT")
    w("=" * 80)
    w(f"  Generated : {now}")
    w(f"  Env file  : {ENV_FILE}")
    w(f"  Timeout   : {REQUEST_TIMEOUT}s per request | Retries: {RETRY_ATTEMPTS}")
    w("-" * 80)
    w(f"  TOTAL MODELS TESTED : {total}")
    w(f"  ✓ UP                : {up}  ({round(up/total*100)}%)")
    w(f"  ✗ DOWN              : {down}  ({round(down/total*100)}%)")
    if skip:
        w(f"  ~ SKIPPED           : {skip}")
    w("=" * 80)

    for platform in platforms:
        p_results = by_platform[platform]
        if not p_results:
            continue

        p_up   = sum(1 for r in p_results if r["status"] == "UP")
        p_down = sum(1 for r in p_results if r["status"] == "DOWN")

        w("")
        w("─" * 80)
        w(f"  PLATFORM: {platform.upper()}")
        w(f"  Models: {len(p_results)} total | ✓ {p_up} up | ✗ {p_down} down")
        w("─" * 80)

        # Split into UP and DOWN groups
        for status_group, label in [("UP", "✓ RUNNING"), ("DOWN", "✗ NOT RUNNING")]:
            group = [r for r in p_results if r["status"] == status_group]
            if not group:
                continue

            w(f"\n  {label} ({len(group)} models):")
            w("  " + "·" * 76)

            # Sort by latency for UP, alphabetically for DOWN
            if status_group == "UP":
                group = sorted(group, key=lambda x: x["latency_s"])
            else:
                group = sorted(group, key=lambda x: x["model_id"])

            for r in group:
                w(f"\n    Model    : {r['model_id']}")
                w(f"    Type     : {r['model_type']}")
                w(f"    Status   : {r['status']}")
                w(f"    Latency  : {r['latency_s']}s")
                w(f"    Detail   : {r['detail']}")
                # Show which classification keys reference this model (deduplicated)
                unique_cats = sorted(set(r["categories"]))
                # Truncate if many categories
                if len(unique_cats) > 6:
                    cat_display = ", ".join(unique_cats[:6]) + f" ... (+{len(unique_cats)-6} more)"
                else:
                    cat_display = ", ".join(unique_cats)
                w(f"    .env keys: {cat_display}")

        w("")

    # ── SUMMARY TABLE ──────────────────────────────────────────
    w("=" * 80)
    w("  QUICK REFERENCE TABLE")
    w("=" * 80)
    w(f"  {'STATUS':<8} {'PLATFORM':<14} {'TYPE':<12} {'LATENCY':>8}  MODEL ID")
    w("  " + "-" * 76)
    for r in sorted(results, key=lambda x: (x["status"] == "UP", x["platform"], x["model_id"])):
        icon = "✓ UP  " if r["status"] == "UP" else "✗ DOWN"
        lat  = f"{r['latency_s']}s" if r["status"] == "UP" else "—"
        w(f"  {icon:<8} {r['platform']:<14} {r['model_type']:<12} {lat:>8}  {r['model_id']}")

    w("")
    w("=" * 80)
    w("  END OF REPORT")
    w("=" * 80)

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MODEL HEALTH CHECK")
    print("  OpenRouter | Groq | Cohere")
    print("=" * 60)

    # Validate keys
    missing_keys = []
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        missing_keys.append("OPENROUTER_API_KEY")
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        missing_keys.append("GROQ_API_KEY")
    if COHERE_API_KEY == "YOUR_COHERE_API_KEY_HERE":
        missing_keys.append("COHERE_API_KEY")

    if missing_keys:
        print(f"\n  ⚠  Missing API keys: {', '.join(missing_keys)}")
        print("     Tests for those platforms will return AUTH errors.")
        print("     Set them in the script or as environment variables.\n")

    # Parse env
    print(f"\n  Parsing: {ENV_FILE}")
    try:
        models = parse_env_file(ENV_FILE)
    except FileNotFoundError as e:
        print(f"\n  ✗ ERROR: {e}")
        print(f"  Make sure '{ENV_FILE}' is in the same directory as this script.")
        return

    # Count by platform
    by_platform = {}
    for info in models.values():
        by_platform[info["platform"]] = by_platform.get(info["platform"], 0) + 1
    for p, count in sorted(by_platform.items()):
        print(f"    {p.upper():14}: {count} unique models")
    print(f"    {'TOTAL':14}: {len(models)} unique models")

    # Run tests
    results = run_all_tests(models)

    # Write report
    print(f"\n  Writing report to: {REPORT_FILE}")
    write_report(results, REPORT_FILE)

    # Terminal summary
    up   = sum(1 for r in results if r["status"] == "UP")
    down = sum(1 for r in results if r["status"] == "DOWN")
    print("\n" + "=" * 60)
    print(f"  DONE — {up} UP / {down} DOWN out of {len(results)} models")
    print(f"  Report saved: {REPORT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()