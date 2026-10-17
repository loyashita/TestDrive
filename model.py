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
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

# --- backend config ----------------------------------------------------------
# MODEL_BACKEND switches between "gemini" (fast, needs API key) and
# "ollama" (local, no internet). Set in .env; default is gemini.
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Retry suffix from PROMPTS.md section 5 — copied exactly, not paraphrased.
# {error} is filled in with what went wrong so the model can self-correct.
RETRY_SUFFIX = (
    "\n\nYour previous response was invalid: {error}\n"
    "Return ONLY a valid JSON object with exactly the required fields "
    "and allowed values. No other text."
)

# Fallback dict returned when all retries fail.
# The "_warning" key signals to callers that this is synthetic, not real.
# Rule: fallback reactions write no memory (checked by rehearsal.py later).
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
    "_warning": "fallback",  # overwritten with details below
}


# --- helpers -----------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers that models add.
    We do this defensively even when json_mode=True, because some models
    ignore the instruction.
    """
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)   # opening fence
    text = re.sub(r"\n?```$", "", text)           # closing fence
    return text.strip()


def _parse_json(text: str) -> dict:
    """Strip fences then parse JSON. Raises ValueError with a clear message."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Keep the raw text short so it fits in a retry prompt
        raise ValueError(f"JSON parse error ({e}). Got: {cleaned[:200]!r}")


# --- backend calls -----------------------------------------------------------

def _call_gemini(prompt: str, image=None, json_mode: bool = True,
                 temperature: float = 0.7) -> str:
    """Send prompt (+optional image bytes) to Gemini. Returns raw text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        temperature=temperature,
        # Tells Gemini to constrain its output to valid JSON — faster and
        # cleaner than hoping it wraps correctly on its own.
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    if image is not None:
        # Detect PNG vs JPEG by magic bytes so callers don't have to care.
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


def _call_ollama(prompt: str, image=None, json_mode: bool = True,
                 temperature: float = 0.7) -> str:
    """Send prompt (+optional image bytes) to Ollama. Returns raw text."""
    import base64
    import ollama

    kwargs = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {"temperature": temperature},
    }
    if json_mode:
        kwargs["format"] = "json"  # Ollama's equivalent of response_mime_type
    if image is not None:
        kwargs["images"] = [base64.b64encode(image).decode()]

    response = ollama.generate(**kwargs)
    return response.response


# --- public API --------------------------------------------------------------

def call_model(
    prompt: str,
    image=None,
    json_mode: bool = True,
    temperature: float = 0.7,
    validate=None,
) -> dict:
    """
    Call the active backend and return a parsed dict.

    Args:
        prompt:      Full prompt string.
        image:       Optional raw image bytes (for multimodal calls).
        json_mode:   Ask the backend to return JSON. True for all structured
                     calls; only False for free-text (not used here yet).
        temperature: 0 = deterministic, 1 = creative.
                     Use ~0.9 for citizens and headline, ~0.2 for extraction
                     and director (we want consistent logic there).
        validate:    Optional function(dict) -> str | None.
                     Return an error string if something is wrong, None if OK.
                     On failure we retry up to 2x with RETRY_SUFFIX attached.

    Returns:
        Parsed dict on success.
        FALLBACK dict (with "_warning" key) if all 3 attempts fail.
        Fallback reactions must NOT write memory — callers check for "_warning".
    """
    _call = _call_gemini if MODEL_BACKEND == "gemini" else _call_ollama

    current_prompt = prompt
    last_error = None

    for attempt in range(3):  # attempt 0 = first try; 1, 2 = retries
        try:
            raw = _call(current_prompt, image=image, json_mode=json_mode,
                        temperature=temperature)
            result = _parse_json(raw)

            if validate is not None:
                error = validate(result)
                if error:
                    raise ValueError(error)

            return result  # all good

        except ValueError as e:
            last_error = str(e)
            if attempt < 2:
                # Send the error back to the model so it can self-correct.
                # This is the retry loop — we don't change the original prompt,
                # we append what went wrong so the model sees its own mistake.
                current_prompt = prompt + RETRY_SUFFIX.format(error=last_error)

    # All 3 attempts failed. Return a flagged fallback so the UI can show "⚠"
    # and rehearsal.py knows not to write this to memory.
    fallback = dict(FALLBACK)
    fallback["_warning"] = (
        f"⚠ model fallback after 3 attempts — last error: {last_error}"
    )
    return fallback


# --- quick self-test ---------------------------------------------------------
# Run:  python model.py           → tests whatever MODEL_BACKEND is in .env
#       python model.py gemini    → force Gemini
#       python model.py ollama    → force Ollama
# Done when: the same prompt returns parsed JSON on both backends.

if __name__ == "__main__":
    import sys

    # Allow overriding the backend from the command line for this test only.
    # This works because call_model reads MODEL_BACKEND as a global at call time.
    if len(sys.argv) > 1 and sys.argv[1] in ("gemini", "ollama"):
        MODEL_BACKEND = sys.argv[1]

    TEST_PROMPT = (
        'Return a JSON object with exactly two fields: '
        '"name" (any made-up name) and "mood" (one word). '
        'Return ONLY the JSON object, nothing else.'
    )

    def _validate_test(d: dict):
        """Check both expected fields are present."""
        missing = [f for f in ("name", "mood") if f not in d]
        if missing:
            return f"Missing fields: {missing}"
        return None

    print(f"Testing backend: {MODEL_BACKEND}")
    result = call_model(TEST_PROMPT, temperature=0.7, validate=_validate_test)

    if "_warning" in result:
        print(f"FAILED: {result['_warning']}")
        sys.exit(1)
    else:
        print(f"PASSED: {result}")
