"""
memory.py — load / save / reset each citizen's state (trust, outlook, memories).

Key concept: state persistence + in-context memory
  The model's weights never change. A citizen "remembers" only because we
  save their history to a JSON file after each drive and paste the last 3
  memories into their next prompt. The memory lives HERE (disk), not in Gemma.

State file: memory/{crowd_id}.json, shape from SPEC §9:
  { "c1": {"trust": 6, "outlook": "...", "memories": [...], "trust_history": [...]}, ... }

GUARDRAILS covered here:
  6.1 — atomic writes (write .tmp, then os.replace)
  6.2 — corrupt file -> back up as .bak, start fresh, return a warning
  6.3 — crowd id is checked against crowds.json before it becomes a file path
  6.4 — history capped: 30 memories, 50 trust points per citizen
"""

import json
import os
from pathlib import Path

# Paths are relative to this file, so it works no matter where you run from.
BASE_DIR = Path(__file__).parent
CROWDS_FILE = BASE_DIR / "crowds.json"
MEMORY_DIR = BASE_DIR / "memory"

MAX_MEMORIES = 30       # GUARDRAILS 6.4
MAX_TRUST_POINTS = 50   # GUARDRAILS 6.4

# How a stored action reads in the "what you remember" prompt block.
ACTION_PHRASES = {
    "keep": "I carried on as before",
    "increase": "I bought more",
    "reduce": "I cut back",
    "switch": "I switched to something else",
    "drop_out": "I dropped out",
}


# --- crowds ------------------------------------------------------------------

def load_crowds() -> list[dict]:
    with open(CROWDS_FILE, encoding="utf-8") as f:
        return json.load(f)["crowds"]


def _state_path(crowd_id: str) -> Path:
    """GUARDRAILS 6.3 — only ids that exist in crowds.json may become a path."""
    known = {c["id"] for c in load_crowds()}
    if crowd_id not in known:
        raise ValueError(f"Unknown crowd id: {crowd_id!r}")
    return MEMORY_DIR / f"{crowd_id}.json"


# --- state -------------------------------------------------------------------

def _initial_citizen_state(citizen: dict) -> dict:
    """Starting values come from crowds.json; the backstory is memory #0."""
    return {
        "trust": citizen["starting_trust"],
        "outlook": citizen["starting_outlook"],
        "memories": [{
            "drive": 0, "decision": "backstory", "action": None,
            "sentiment": None, "note": citizen["backstory_memory"],
        }],
        "trust_history": [citizen["starting_trust"]],
    }


def _initial_state(crowd: dict) -> dict:
    return {c["id"]: _initial_citizen_state(c) for c in crowd["citizens"]}


def _citizen_state_ok(s) -> bool:
    return (
        isinstance(s, dict)
        and isinstance(s.get("trust"), (int, float))
        and isinstance(s.get("outlook"), str)
        and isinstance(s.get("memories"), list)
        and isinstance(s.get("trust_history"), list)
    )


def load_state(crowd: dict) -> tuple[dict, str | None]:
    """
    Return (state, warning). warning is None unless the file was corrupt.
    A missing file just means "first drive": return the starting state.
    """
    path = _state_path(crowd["id"])
    if not path.exists():
        return _initial_state(crowd), None

    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict) or not all(
            _citizen_state_ok(s) for s in state.values()
        ):
            raise ValueError("unexpected structure")
    except (OSError, ValueError):   # JSONDecodeError is a ValueError
        # GUARDRAILS 6.2 — keep the broken file for inspection, never crash.
        os.replace(path, path.with_name(path.name + ".bak"))
        return _initial_state(crowd), (
            "Memory file was unreadable; backed it up as .bak and started fresh."
        )

    # Crowd may have gained a citizen since the file was written.
    for c in crowd["citizens"]:
        state.setdefault(c["id"], _initial_citizen_state(c))
    return state, None


def save_state(crowd: dict, state: dict) -> None:
    """GUARDRAILS 6.1 — write to .tmp first so a crash can't half-write memory."""
    path = _state_path(crowd["id"])
    MEMORY_DIR.mkdir(exist_ok=True)

    for s in state.values():    # GUARDRAILS 6.4
        s["memories"] = s["memories"][-MAX_MEMORIES:]
        s["trust_history"] = s["trust_history"][-MAX_TRUST_POINTS:]

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)       # atomic: old file or new file, never half


def reset_state(crowd: dict) -> dict:
    """Wipe memories: back to starting trust, outlook and backstory."""
    state = _initial_state(crowd)
    save_state(crowd, state)
    return state


def next_drive_number(state: dict) -> int:
    """Derived from the memories themselves, so no separate counter can drift."""
    last = max((m["drive"] for s in state.values() for m in s["memories"]), default=0)
    return last + 1


# --- prompt formatting -------------------------------------------------------

def format_memories(memories: list[dict], n: int = 3) -> str:
    """
    The {memories} block of CITIZEN_REACT: last n memories, oldest first.
    Only n go in, so the prompt stays short and recent history dominates.
    """
    recent = memories[-n:]
    if not recent:
        return "- Nothing yet."

    lines = []
    for m in recent:
        if m["drive"] == 0:
            lines.append(f"- Before any of this: {m['note']}")
        else:
            did = ACTION_PHRASES.get(m["action"], "I reacted")
            lines.append(f"- Drive {m['drive']} ({m['decision']}): {did}. \"{m['note']}\"")
    return "\n".join(lines)
