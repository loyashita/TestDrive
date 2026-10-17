"""
check_setup.py -- Layer 0 verification.
Run with: python check_setup.py
All three checks must pass before moving to Layer 1.
"""

import json
import os
import sys


def check_crowds():
    """Load crowds.json and print the cast list."""
    with open("crowds.json", encoding="utf-8") as f:
        data = json.load(f)
    crowd = data["crowds"][0]
    print(f"[OK] crowds.json loaded -- crowd: '{crowd['title']}', {len(crowd['citizens'])} citizens")
    for c in crowd["citizens"]:
        print(f"     {c['id']} {c['name']}, {c['age']}, budget Rs{c['monthly_budget']}")
    return True


def check_env():
    """Check that .env exists and GEMINI_API_KEY is set (non-placeholder)."""
    if not os.path.exists(".env"):
        print("[X]  .env not found. Copy .env.example to .env and fill in your key.")
        return False
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your_key_here":
        print("[X]  GEMINI_API_KEY is missing or still the placeholder in .env.")
        return False
    model = os.getenv("GEMINI_MODEL", "(not set)")
    print(f"[OK] GEMINI_API_KEY found (ends ...{key[-4:]}), model: {model}")
    return True


def check_ollama():
    """Ping Ollama and confirm gemma4:e4b is available."""
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        target = "gemma4:e4b"
        if any(target in n for n in names):
            print(f"[OK] Ollama: {target} is available")
        else:
            print(f"[!]  Ollama is running but {target} is not pulled yet.")
            print(f"     Run: ollama pull {target}")
            print(f"     Available models: {names or '(none)'}")
        return True
    except Exception as e:
        print(f"[X]  Ollama not reachable: {e}")
        print("     Is Ollama running? Start it with: ollama serve")
        return False


if __name__ == "__main__":
    print("-- Layer 0 check --")
    ok_crowds = check_crowds()
    print()
    ok_env = check_env()
    print()
    ok_ollama = check_ollama()
    print()
    if ok_crowds and ok_env and ok_ollama:
        print("All checks passed. Ready for Layer 1.")
    else:
        print("Fix the issues above, then run this again.")
        sys.exit(1)
