"""
rehearsal.py — orchestration: one "test drive" of a decision on a crowd.

This is where the pieces meet:
    memory.py  (what each person remembers)
 -> engine.py  (how the decision LANDS: perception, pure Python)
 -> model.py   (what each person SAYS and DOES: Gemma)
 -> engine.py  (what it COSTS: metrics, trust, pure Python)
 -> memory.py  (save the new history)

Gemma decides, Python counts. The model never sees a formula and never
computes money or trust; it is *handed* the perception summary as fact.

Run:
    python rehearsal.py              # three drives back to back
    python rehearsal.py reset        # wipe memories first, then three drives
    python rehearsal.py 2            # only drive 2 (state loads from disk)
    python rehearsal.py ollama       # force a backend (or: gemini)

GUARDRAILS covered here:
  2.5 — citizen_id always overwritten by code, never trusted from the model
  3.3 — length caps on model text (truncate, don't reject)
  4.1 — fallback reactions write no memory and don't move trust
  4.2 — validator: required fields, allowed enums, reduce_level fix, sentiment clamp
  4.6 — 3+ fallbacks in one drive -> drive is "unreliable" and is NOT saved
"""

import sys

import engine
import memory
import model
import prompts

ACTIONS = {"keep", "increase", "reduce", "switch", "drop_out"}
VOICES = {"silent", "grumble", "complain_publicly", "defend", "rally_others"}
REDUCE_LEVELS = {"light", "half", "most"}
REQUIRED_FIELDS = [
    "action", "reduce_level", "voice", "sentiment", "quote",
    "inner_thought", "reason", "memory_note", "updated_outlook",
]
SHORT_FIELDS = ("quote", "inner_thought", "memory_note")   # <= 160 chars
LONG_FIELDS = ("reason", "updated_outlook")                # <= 240 chars

UNRELIABLE_AT = 3   # GUARDRAILS 4.6


# --- validation --------------------------------------------------------------

def _make_validator(citizen_id: str):
    """
    Build the function call_model uses to check a reaction.
    Returns an error string (-> retry with that error) or None (OK).
    Things that are safely fixable are fixed in place instead of retried.
    """
    def validate(r) -> str | None:
        if not isinstance(r, dict):
            return "Response must be a JSON object."
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            return f"Missing fields: {missing}"

        if r["action"] not in ACTIONS:
            return f"action must be one of {sorted(ACTIONS)}, got {r['action']!r}"
        if r["voice"] not in VOICES:
            return f"voice must be one of {sorted(VOICES)}, got {r['voice']!r}"

        try:
            r["sentiment"] = int(max(-3, min(3, round(float(r["sentiment"])))))
        except (TypeError, ValueError):
            return f"sentiment must be an integer -3 to 3, got {r['sentiment']!r}"

        # reduce_level is set if and only if action == "reduce"
        if r["action"] == "reduce":
            if r["reduce_level"] not in REDUCE_LEVELS:
                r["reduce_level"] = "half"
        else:
            r["reduce_level"] = None

        for f in SHORT_FIELDS + LONG_FIELDS:
            if not isinstance(r[f], str):
                return f"{f} must be a string."
            r[f] = r[f].strip()[: 160 if f in SHORT_FIELDS else 240]
        if not r["quote"] or not r["inner_thought"]:
            return "quote and inner_thought must not be empty."

        r["citizen_id"] = citizen_id   # GUARDRAILS 2.5
        return None

    return validate


# --- one citizen -------------------------------------------------------------

def _price_text(p) -> str:
    return "n/a" if p is None else f"Rs{p:g}"


def react(citizen: dict, state: dict, perception: dict, decision: dict, crowd: dict) -> dict:
    """
    Fill the CITIZEN_REACT prompt, call the model, return a validated reaction.
    If the model fails 3 times the reaction carries a "_warning" key.
    """
    cs = state[citizen["id"]]
    prompt = prompts.CITIZEN_REACT.format(
        name=citizen["name"], age=citizen["age"], role=citizen["role"],
        voice_style=citizen["voice_style"], personality=citizen["personality"],
        core_values="; ".join(citizen["core_values"]),
        constraints="; ".join(citizen["constraints"]),
        stakes=citizen["stakes"], rule_attitude=citizen["rule_attitude"],
        social_role=citizen["social_role"],
        triggers="; ".join(citizen["triggers"]),
        soft_spots="; ".join(citizen["soft_spots"]),
        secret=citizen["secret"],   # always in the citizen's OWN prompt
        contradiction=citizen["contradiction"],
        line_wont_cross=citizen["line_wont_cross"],
        what_wins_them_back=citizen["what_wins_them_back"],
        decision_maker=decision["decision_maker"],
        trust=cs["trust"], outlook=cs["outlook"],
        memories=memory.format_memories(cs["memories"]),
        decision_title=decision["title"],
        decision_description=decision["description"],
        old_price=_price_text(decision.get("old_price")),
        new_price=_price_text(decision.get("new_price")),
        unit=decision.get("unit") or "n/a",
        baseline_qty=perception.get("baseline_qty", "n/a (not a purchase)"),
        perception_summary=perception["summary"],
        citizen_id=citizen["id"],
    )
    reaction = model.call_model(
        prompt,
        temperature=model.TEMP_CITIZEN,
        validate=_make_validator(citizen["id"]),
    )
    reaction["citizen_id"] = citizen["id"]   # also covers the fallback dict
    return reaction


# --- one drive ---------------------------------------------------------------

def decision_label(decision: dict) -> str:
    """Short label stored in memories, e.g. 'Fest pass Rs349 -> Rs499'."""
    old, new = decision.get("old_price"), decision.get("new_price")
    if engine._is_valid_price(old) and engine._is_valid_price(new):
        return f"{decision['title']} {_price_text(old)} -> {_price_text(new)}"
    return decision["title"]


def run_drive(crowd: dict, decision: dict, progress=None) -> dict:
    """
    Run one full drive and (unless unreliable) save the new state.
    progress: optional function(i, total, name) so a UI can show "Ananya is thinking".
    """
    state, warning = memory.load_state(crowd)
    drive_number = memory.next_drive_number(state)
    citizens = crowd["citizens"]

    decision = dict(decision)
    decision["decision_maker"] = decision.get("decision_maker") or crowd["decision_maker_default"]

    trusts_before = [state[c["id"]]["trust"] for c in citizens]
    perceptions = [engine.perceive(c, decision, t) for c, t in zip(citizens, trusts_before)]

    reactions = []
    for i, (c, p) in enumerate(zip(citizens, perceptions)):
        if progress:
            progress(i + 1, len(citizens), c["name"])
        reactions.append(react(c, state, p, decision, crowd))   # sequential: GUARDRAILS 4.5

    fallbacks = sum(1 for r in reactions if "_warning" in r)
    unreliable = fallbacks >= UNRELIABLE_AT
    metrics = engine.compute_metrics(citizens, reactions, perceptions, decision, trusts_before)

    trusts_after = list(trusts_before)
    if not unreliable:
        label = decision_label(decision)
        for i, (c, r) in enumerate(zip(citizens, reactions)):
            if "_warning" in r:
                continue   # GUARDRAILS 4.1: fallbacks change nothing
            cs = state[c["id"]]
            new_trust = engine.update_trust(cs["trust"], r["sentiment"], r["action"], r["voice"],
                                       is_price=perceptions[i]["is_price"])
            cs["trust"] = new_trust
            cs["trust_history"].append(new_trust)
            cs["outlook"] = r["updated_outlook"] or cs["outlook"]
            cs["memories"].append({
                "drive": drive_number, "decision": label, "action": r["action"],
                "sentiment": r["sentiment"], "note": r["memory_note"],
            })
            trusts_after[i] = new_trust
        memory.save_state(crowd, state)

    metrics["avg_trust_after"] = round(sum(trusts_after) / len(trusts_after), 2)
    return {
        "drive_number": drive_number,
        "decision": decision,
        "citizens": citizens,
        "perceptions": perceptions,
        "reactions": reactions,
        "trusts_before": trusts_before,
        "trusts_after": trusts_after,
        "metrics": metrics,
        "fallbacks": fallbacks,
        "unreliable": unreliable,
        "saved": not unreliable,
        "memory_warning": warning,
    }


# --- Layer 4: photo / text -> structured decision ----------------------------

MAX_TEXT_CHARS = 500        # GUARDRAILS 2.3
MAX_IMAGE_SIDE = 1280       # GUARDRAILS 2.4


def prepare_image(raw: bytes) -> bytes:
    """
    GUARDRAILS 2.4 — shrink to max 1280px and re-encode as JPEG, in memory only.
    Re-encoding also means we only ever forward a real decoded image.
    """
    import io
    from PIL import Image   # already installed as a Streamlit dependency

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _clean_price(p):
    """Number in (0, 100000] or None (GUARDRAILS 5.1)."""
    if isinstance(p, bool):
        return None
    if isinstance(p, str):
        p = p.replace("Rs", "").replace("₹", "").replace(",", "").strip()
    try:
        return float(p) if engine._is_valid_price(p) else None
    except (TypeError, ValueError):
        return None


# Used only if the model blocks but forgets to write its own witty line.
WITTY_FALLBACK = {
    "none": "",
    "hateful": "Nice try, but this crowd is here to react to decisions, not to be mean to people. Rephrase it as an actual decision?",
    "harmful": "That one's off the menu: Test Drive only rehearses decisions nobody gets hurt by. Got a policy or a price instead?",
    "cheating": "Ha, I see what you did there. Instructions hidden in a poster don't work on this crowd. Describe the decision itself!",
}
CANT_READ = "That one slipped past me. I couldn't find a decision in it. Try typing it in plain words, like 'Library closes at 11pm'."


def _validate_decision(d) -> str | None:
    if not isinstance(d, dict):
        return "Response must be a JSON object."
    missing = [f for f in ("title", "description", "old_price", "new_price", "unit",
                           "decision_maker", "block") if f not in d]
    if missing:
        return f"Missing fields: {missing}"
    for f in ("title", "description", "unit"):
        if not isinstance(d[f], str) or not d[f].strip():
            return f"{f} must be a non-empty string."
        d[f] = d[f].strip()[:160]
    d["old_price"] = _clean_price(d["old_price"])
    d["new_price"] = _clean_price(d["new_price"])
    d["decision_maker"] = d["decision_maker"].strip()[:100] if isinstance(d["decision_maker"], str) and d["decision_maker"].strip() else None
    if d["block"] not in ("none", "hateful", "harmful", "cheating"):
        return 'block must be one of "none", "hateful", "harmful", "cheating".'
    # Code, not the model, has the last word: only cheating and clearly harmful
    # input are blocked. "hateful" (and any mislabelled group-differentiated
    # policy, e.g. gender-specific curfews) always runs; the citizen prompt
    # itself forbids slurs and demeaning output (GUARDRAILS 3.2).
    d["safe"] = d["block"] not in ("cheating", "harmful")
    d["safety_reason"] = str(d.get("witty_reply") or "")[:240] or WITTY_FALLBACK[d["block"]]
    return None


def extract_decision(image: bytes | None = None, text: str = "") -> tuple[dict | None, str | None]:
    """
    Gemma (multimodal) turns a poster photo and/or typed text into a Decision.
    Returns (decision, error). Never guesses a price: the prompt forbids it,
    and _validate_decision drops anything that isn't a sane number.
    """
    text = (text or "").strip()[:MAX_TEXT_CHARS]
    if image is None and not text:
        return None, "Give me a photo or some text."
    if image is not None:
        try:
            image = prepare_image(image)
        except Exception:
            return None, "Couldn't read that image. Try a PNG or JPG."

    result = model.call_model(
        prompts.DECISION_EXTRACT.format(user_text=text or "(none)"),
        image=image, temperature=model.TEMP_EXTRACT, validate=_validate_decision,
    )
    if "_warning" in result:
        return None, CANT_READ
    return result, None


# --- Layer 6: director's notes and headline ----------------------------------

def _reactions_summary(result: dict) -> str:
    lines = []
    rows = zip(result["citizens"], result["perceptions"], result["reactions"],
               result["trusts_before"], result["trusts_after"])
    for c, p, r, t0, t1 in rows:
        if "_warning" in r:
            lines.append(f"{c['id']} {c['name']} | (no reaction: model failed)")
            continue
        felt = f"{p['money_feel']} / {p['fairness_feel']}" if p["is_price"] else "not about money"
        level = f" ({r['reduce_level']})" if r["reduce_level"] else ""
        lines.append(
            f"{c['id']} {c['name']} (trust {t0:.1f} → {t1:.1f}) | felt: {felt} | "
            f"SAID: \"{r['quote']}\" | THOUGHT: \"{r['inner_thought']}\" | "
            f"DID: {r['action']}{level}, {r['voice']}, {r['sentiment']} | WHY: {r['reason']}"
        )
    return "\n".join(lines)


def _metrics_summary(m: dict) -> str:
    parts = []
    if m["revenue_change_pct"] is not None:
        parts.append(f"revenue {m['revenue_change_pct']:+.1f}%")
        parts.append(f"elasticity {m['elasticity']} ({m['elasticity_label']})")
    parts.append(f"exit {m['exit_count']}, voice {m['voice_count']}, loyalty {m['loyalty_count']}, "
                 f"grudging {m['grudging_count']}, say-do gaps {m['say_do_gaps']}")
    parts.append(f"avg trust {m['avg_trust']} → {m['avg_trust_after']}")
    return "; ".join(parts)


def direct(result: dict) -> dict | None:
    """
    Director's notes. It is handed computed numbers (not asked to calculate)
    and sees what happened, never the secrets, so its tweak must be something
    the decision-maker could actually do. Returns None if the model fails.
    """
    import json
    ids = {c["id"] for c in result["citizens"]}
    d = result["decision"]
    original = {k: d.get(k) for k in ("title", "description", "old_price", "new_price", "unit")}

    def validate(n) -> str | None:
        if not isinstance(n, dict):
            return "Response must be a JSON object."
        for f in ("hurt_most", "missed_objection", "hidden_issue", "trust_warning", "tweak", "tweaked_decision"):
            if f not in n:
                return f"Missing field: {f}"
        if n["hurt_most"] not in ids:
            n["hurt_most"] = result["metrics"]["hurt_most"]   # Python's answer wins
        for f in ("missed_objection", "hidden_issue", "trust_warning", "tweak"):
            n[f] = str(n[f]).strip()[:240]
        t = n["tweaked_decision"]
        if not isinstance(t, dict) or not all(k in t for k in ("title", "description", "new_price")):
            return "tweaked_decision needs title, description, old_price, new_price, unit."
        n["tweaked_decision"] = {
            "title": str(t["title"])[:160], "description": str(t["description"])[:240],
            "old_price": original["old_price"],            # always the original
            "new_price": _clean_price(t["new_price"]),
            "unit": original["unit"],
            "decision_maker": d["decision_maker"],
        }
        return None

    notes = model.call_model(
        prompts.DIRECTOR_NOTES.format(
            decision_json=json.dumps(original, ensure_ascii=False),
            reactions_summary=_reactions_summary(result),
            metrics_summary=_metrics_summary(result["metrics"]),
        ),
        temperature=model.TEMP_EXTRACT, validate=validate,
    )
    return None if "_warning" in notes else notes


def headline(result: dict) -> str | None:
    """One-line newspaper headline (P2). Returns None on any failure."""
    ok = [(c, r, t0, t1) for c, r, t0, t1 in zip(
        result["citizens"], result["reactions"], result["trusts_before"], result["trusts_after"])
        if "_warning" not in r]
    if not ok:
        return None
    big = max(ok, key=lambda x: abs(x[3] - x[2]))
    loud = max(ok, key=lambda x: abs(x[1]["sentiment"]))

    def validate(h) -> str | None:
        if not isinstance(h, dict) or not isinstance(h.get("headline"), str) or not h["headline"].strip():
            return 'Need {"headline": "..."}'
        h["headline"] = h["headline"].strip()[:140]
        return None

    out = model.call_model(
        prompts.HEADLINE.format(
            decision_title=result["decision"]["title"], drive_number=result["drive_number"],
            metrics_summary=_metrics_summary(result["metrics"]),
            biggest_trust_change=f"{big[0]['name']} {big[2]:g} → {big[3]:g}",
            best_quote=loud[1]["quote"], quote_name=loud[0]["name"]),
        temperature=model.TEMP_HEADLINE, validate=validate)
    return None if "_warning" in out else out["headline"]


# --- terminal output ---------------------------------------------------------

def print_drive(result: dict) -> None:
    d = result["decision"]
    print("\n" + "=" * 72)
    print(f"DRIVE {result['drive_number']}: {decision_label(d)}")
    print(f"  {d['description']}  (decided by {d['decision_maker']})")
    print("=" * 72)
    if result["memory_warning"]:
        print(f"[!] {result['memory_warning']}")

    rows = zip(result["citizens"], result["perceptions"], result["reactions"],
               result["trusts_before"], result["trusts_after"])
    for c, p, r, t0, t1 in rows:
        arrow = "▲" if t1 > t0 else "▼" if t1 < t0 else "="
        print(f"\n{c['name']}   trust {t0:g} -> {t1:g} {arrow}")
        if p["is_price"]:
            print(f"  LANDED:  money: {p['money_feel']} | fairness: {p['fairness_feel']}")
        else:
            print(f"  LANDED:  {p['summary']}")
        if "_warning" in r:
            print(f"  [!] fallback reaction, nothing saved: {r['_warning']}")
            continue
        level = f" ({r['reduce_level']})" if r["reduce_level"] else ""
        print(f"  SAID:    \"{r['quote']}\"")
        print(f"  THOUGHT: \"{r['inner_thought']}\"")
        print(f"  DID:     {r['action']}{level}, {r['voice']}, sentiment {r['sentiment']:+d}")
        print(f"  WHY:     {r['reason']}")

    m = result["metrics"]
    print("\n--- numbers (Python) ---")
    if m["revenue_change_pct"] is not None:
        print(f"  revenue {m['old_revenue']:g} -> {m['new_revenue']:g} "
              f"({m['revenue_change_pct']:+.1f}%) | elasticity {m['elasticity']} ({m['elasticity_label']})")
    print(f"  exit {m['exit_count']} | voice {m['voice_count']} | loyalty {m['loyalty_count']} "
          f"| grudging {m['grudging_count']} | say-do gaps {m['say_do_gaps']}")
    print(f"  avg sentiment {m['avg_sentiment']} | avg trust {m['avg_trust']} -> {m['avg_trust_after']}"
          f" | hurt most: {m['hurt_most']}")
    if result["unreliable"]:
        print(f"\n[!] {result['fallbacks']} of 6 reactions fell back. Drive NOT saved. "
              "Retry, or switch backend.")


# --- the three demo drives ---------------------------------------------------

DEMO_DRIVES = [
    {"title": "Fest pass", "description": "Fest pass goes from Rs349 to Rs499",
     "old_price": 349, "new_price": 499, "unit": "fest pass",
     "decision_maker": "the fest organising committee"},
    {"title": "Fest pass", "description": "Next year the fest pass goes from Rs499 to Rs549",
     "old_price": 499, "new_price": 549, "unit": "fest pass",
     "decision_maker": "the fest organising committee"},
    {"title": "Library closing time",
     "description": "Library closes at 11pm instead of 2am",
     "old_price": None, "new_price": None, "unit": "library hours",
     "decision_maker": "the administration"},
]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):   # Windows console + ▲ ▼
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    for a in args:
        if a in ("gemini", "ollama"):
            model.MODEL_BACKEND = a
    crowd = memory.load_crowds()[0]

    if "reset" in args:
        memory.reset_state(crowd)
        print("Memories reset.")

    only = [int(a) for a in args if a.isdigit()]
    chosen = [DEMO_DRIVES[n - 1] for n in only] if only else DEMO_DRIVES
    print(f"Backend: {model.MODEL_BACKEND}")
    for dec in chosen:
        print_drive(run_drive(crowd, dec, progress=lambda i, n, name: print(f"  ({i}/{n}) {name} is thinking...")))
