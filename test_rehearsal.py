"""
test_rehearsal.py — offline checks for memory.py + rehearsal.py (no model, no network).

A fake call_model stands in for Gemma, and memory is written to a temp folder.
Run with: python test_rehearsal.py
"""

import json
import sys
import tempfile
from pathlib import Path

import memory
import model
import rehearsal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

memory.MEMORY_DIR = Path(tempfile.mkdtemp())
crowd = memory.load_crowds()[0]
DECISION = rehearsal.DEMO_DRIVES[0]
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")


def fake_reaction(prompt, **kw):
    """Behaves like call_model: runs the real validator on a deliberately messy reply."""
    raw = {"citizen_id": "WRONG", "action": "reduce", "reduce_level": None, "voice": "grumble",
           "sentiment": "-2", "quote": "q" * 500, "inner_thought": "t", "reason": "r",
           "memory_note": "n", "updated_outlook": "o"}
    err = kw["validate"](raw)
    return raw if err is None else dict(model.FALLBACK, _warning=err)


print("Drive 1 with a messy-but-fixable model reply")
model.call_model = fake_reaction
r1 = rehearsal.run_drive(crowd, DECISION)
react0 = r1["reactions"][0]
check("citizen_id overwritten (2.5)", react0["citizen_id"] == "c1")
check("sentiment coerced to int", react0["sentiment"] == -2)
check("reduce with no level -> 'half' (4.2)", react0["reduce_level"] == "half")
check("quote truncated to 160 (3.3)", len(react0["quote"]) == 160)
check("drive saved", r1["saved"] and r1["drive_number"] == 1)

print("State survives a 'restart' (fresh load from disk)")
state, warn = memory.load_state(crowd)
check("no warning", warn is None)
check("trust moved: Ananya 6 -> 5", state["c1"]["trust"] == 5.0)
check("memories = backstory + drive 1", len(state["c1"]["memories"]) == 2)
check("drive counter advanced", memory.next_drive_number(state) == 2)
check("memory text shows drive 1", "Drive 1" in memory.format_memories(state["c1"]["memories"]))

print("Drive 2 sees drive 1")
seen = {}
def spy(prompt, **kw):
    seen.setdefault("prompt", prompt)
    return fake_reaction(prompt, **kw)
model.call_model = spy
rehearsal.run_drive(crowd, rehearsal.DEMO_DRIVES[1])
check("drive 1 memory is inside drive 2's prompt", "Drive 1 (Fest pass" in seen["prompt"])
check("secret is in the citizen's own prompt", "sneakers" in seen["prompt"])
check("decision wrapped in USER_INPUT (2.1)", "<<<USER_INPUT" in seen["prompt"])

print("Unreliable drive (3+ fallbacks) is not saved (4.6)")
before = memory.load_state(crowd)[0]["c1"]["trust"]
model.call_model = lambda prompt, **kw: dict(model.FALLBACK, _warning="boom")
bad = rehearsal.run_drive(crowd, DECISION)
check("flagged unreliable", bad["unreliable"] and not bad["saved"])
check("trust unchanged on disk", memory.load_state(crowd)[0]["c1"]["trust"] == before)

print("Non-price drive")
model.call_model = fake_reaction
r3 = rehearsal.run_drive(crowd, rehearsal.DEMO_DRIVES[2])
check("no revenue for non-price", r3["metrics"]["revenue_change_pct"] is None)
check("prompt filled without crashing", r3["saved"])

print("Corrupt file -> .bak + fresh start (6.2); reset (wipe); bad id (6.3)")
path = memory.MEMORY_DIR / "bits_goa.json"
path.write_text("{not json", encoding="utf-8")
state, warn = memory.load_state(crowd)
check("warning returned", warn is not None)
check("backup written", (memory.MEMORY_DIR / "bits_goa.json.bak").exists())
check("fresh starting trust", state["c1"]["trust"] == 6)
memory.reset_state(crowd)
check("reset restarts at drive 1", memory.next_drive_number(memory.load_state(crowd)[0]) == 1)
try:
    memory._state_path("../evil")
    check("unknown crowd id rejected", False)
except ValueError:
    check("unknown crowd id rejected", True)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
