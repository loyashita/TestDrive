"""
model.py — the only file in this project that talks to a language model.

What it does:
  call_model() sends a prompt (and optionally an image) to either Gemini or
  Ollama, strips any markdown fences, parses the JSON response, and—if you
  give it a validate function—retries up to twice when the output is wrong.

Key concept: Structured output / why we can't trust model JSON
  Models are text predictors. Even when told to return JSON they sometimes
  wrap it in ```json fences, leave trailing commas, miss required fields, or
  use wrong enum values. We always parse and validate; we never assume the
  output is correct. That's what the retry loop is for.

GUARDRAILS retrofitted here:
  1.2 — API errors are caught and re-raised without exposing the key.
  4.3 — Timeouts: 60 s Gemini, 180 s Ollama.
  4.4 — 429 rate-limit: back off 2 s then 5 s before giving up.
  4.7 — Temperature defaults: 0.9 citizens/headline, 0.2 extraction/director.
"""

import concurrent.futures
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# --- backend config ----------------------------------------------------------
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# GUARDRAILS 4.3 — timeouts per backend (seconds)
GEMINI_TIMEOUT = 60
OLLAMA_TIMEOUT = 180

# GUARDRAILS 4.7 — temperature presets (callers pick the right one)
TEMP_CITIZEN = 0.9    # creative, in-character
TEMP_EXTRACT = 0.2    # deterministic: extraction, director, validation-sensitive
TEMP_HEADLINE = 0.9

# Retry suffix from PROMPTS.md §5 — copied exactly.
RETRY_SUFFIX = (
    "\n\nYour previous response was invalid: {error}\n"
    "Return ONLY a valid JSON object with exactly the required fields "
    "and allowed values. No other text."
)

# Fallback returned when all retries fail.
# The "_warning" key signals: fallback reactions must NOT write memory
# or update trust (checked by rehearsal.py).
FALLBACK = {
    "action": "keep",
    "reduce_level": None,
    "voice": "silent",
    "sentiment": 0,
    "quote": "",
    "inner_thought": "",
    "reason": "",
    "memory_note": "",
    "updated_outlook": "",
    "_warning": "fallback",
}


# --- helpers -----------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that models add even when told not to."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    """Strip fences then parse JSON. Raises ValueError with a clear message."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error ({e}). Got: {cleaned[:200]!r}")


def _run_with_timeout(fn, timeout_s, *args, **kwargs):
    """
    Run fn(*args, **kwargs) in a thread and raise TimeoutError if it takes
    longer than timeout_s seconds. Cross-platform (no signal tricks needed).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Model call timed out after {timeout_s}s")


# --- backend calls -----------------------------------------------------------

def _gemini_inner(prompt, image, json_mode, temperature):
    """The actual Gemini call — wrapped by _call_gemini for timeout + error handling."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    if image is not None:
        mime = "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"
        parts = [types.Part.from_bytes(data=image, mime_type=mime), prompt]
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=parts, config=config
        )
    else:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
    return response.text


def _call_gemini(prompt, image=None, json_mode=True, temperature=TEMP_CITIZEN):
    """
    Send prompt to Gemini with a 60 s timeout.
    GUARDRAILS 1.2: catches exceptions and re-raises without exposing the key.
    GUARDRAILS 4.4: raises RateLimitError on 429 so call_model can back off.
    """
    try:
        return _run_with_timeout(
            _gemini_inner, GEMINI_TIMEOUT, prompt, image, json_mode, temperature
        )
    except TimeoutError:
        raise  # re-raise as-is for call_model to count as a failed attempt
    except Exception as e:
        msg = str(e)
        # GUARDRAILS 4.4 — detect rate limit
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            raise RateLimitError("Gemini rate-limited (429). Switch to Ollama or wait.")
        # GUARDRAILS 1.2 — never expose the raw error (it may contain headers/key info)
        raise ValueError(f"Gemini API error — check key/quota. ({type(e).__name__})")


def _ollama_inner(prompt, image, json_mode, temperature):
    """The actual Ollama call — wrapped by _call_ollama for timeout + error handling."""
    import base64
    import ollama

    kwargs = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {"temperature": temperature},
    }
    if json_mode:
        kwargs["format"] = "json"
    if image is not None:
        kwargs["images"] = [base64.b64encode(image).decode()]

    response = ollama.generate(**kwargs)
    return response.response


def _call_ollama(prompt, image=None, json_mode=True, temperature=TEMP_CITIZEN):
    """Send prompt to Ollama with a 180 s timeout."""
    try:
        return _run_with_timeout(
            _ollama_inner, OLLAMA_TIMEOUT, prompt, image, json_mode, temperature
        )
    except TimeoutError:
        raise
    except Exception as e:
        raise ValueError(f"Ollama error — is it running? ({type(e).__name__})")


class RateLimitError(Exception):
    """Raised on HTTP 429 / RESOURCE_EXHAUSTED so call_model can back off."""


# --- public API --------------------------------------------------------------

def call_model(
    prompt: str,
    image=None,
    json_mode: bool = True,
    temperature: float = TEMP_CITIZEN,
    validate=None,
) -> dict:
    """
    Call the active backend and return a parsed dict.

    Args:
        prompt:      Full prompt string.
        image:       Optional raw image bytes (for multimodal calls).
        json_mode:   Ask the backend to return JSON.
        temperature: Use TEMP_CITIZEN (0.9) for reactions/headline,
                     TEMP_EXTRACT (0.2) for extraction and director.
        validate:    Optional function(dict) -> str | None.
                     Return an error string if invalid, None if OK.
                     On failure: retry ×2 with RETRY_SUFFIX, then fallback.

    Returns:
        Parsed dict on success.
        FALLBACK dict (with "_warning") if all 3 attempts fail.
        Callers must check for "_warning" and skip memory/trust updates.
    """
    _call = _call_gemini if MODEL_BACKEND == "gemini" else _call_ollama

    current_prompt = prompt
    last_error = None
    backoff_delays = [2, 5]  # GUARDRAILS 4.4 — back-off schedule for 429

    for attempt in range(3):
        try:
            raw = _call(current_prompt, image=image, json_mode=json_mode,
                        temperature=temperature)
            result = _parse_json(raw)

            if validate is not None:
                error = validate(result)
                if error:
                    raise ValueError(error)

            return result  # success

        except RateLimitError as e:
            # GUARDRAILS 4.4 — wait then retry; after 2 waits, fall through to fallback
            last_error = str(e)
            if attempt < 2:
                delay = backoff_delays[attempt]
                print(f"[rate-limit] {e} — waiting {delay}s...")
                time.sleep(delay)
                # Don't append RETRY_SUFFIX for rate-limit; the prompt was fine

        except (ValueError, TimeoutError) as e:
            last_error = str(e)
            if attempt < 2:
                current_prompt = prompt + RETRY_SUFFIX.format(error=last_error)

    # All 3 attempts failed — flagged fallback, caller must not write memory
    fallback = dict(FALLBACK)
    fallback["_warning"] = f"⚠ model fallback after 3 attempts — {last_error}"
    return fallback


# --- quick self-test ---------------------------------------------------------
# python model.py           → uses MODEL_BACKEND from .env
# python model.py gemini    → force Gemini
# python model.py ollama    → force Ollama

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("gemini", "ollama"):
        MODEL_BACKEND = sys.argv[1]

    TEST_PROMPT = (
        'Return a JSON object with exactly two fields: '
        '"name" (any made-up name) and "mood" (one word). '
        'Return ONLY the JSON object, nothing else.'
    )

    def _validate_test(d: dict):
        missing = [f for f in ("name", "mood") if f not in d]
        return f"Missing fields: {missing}" if missing else None

    print(f"Testing backend: {MODEL_BACKEND}")
    result = call_model(TEST_PROMPT, temperature=TEMP_CITIZEN, validate=_validate_test)

    if "_warning" in result:
        print(f"FAILED: {result['_warning']}")
        sys.exit(1)
    else:
        print(f"PASSED: {result}")
