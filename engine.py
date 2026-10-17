"""
engine.py — pure Python perception, quantities, metrics, and trust.

NO model calls live here. "Gemma decides, Python counts."
  - Python works out how a decision LANDS (perception) and what it COSTS
    (quantities, revenue, elasticity, trust).
  - Gemma works out what people SAY and DO about it.

Every formula matches SPEC §10 exactly.
Each comment names the behavioural-economics concept it models.

GUARDRAILS covered here:
  5.1 — Price sanity (0 < price ≤ 100 000).
  5.2 — Division-by-zero guards throughout.
  5.3 — old_price == new_price → "no change".
  5.5 — clamp() helper; round for display.
  5.6 — Unit matching is case-insensitive, falls back to "default".
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Keep a value inside [lo, hi]. Used for trust (0–10) and sentiments."""
    return max(lo, min(hi, value))


def _get_baseline_qty(citizen: dict, unit: str) -> float:
    """
    SPEC §10.1 / GUARDRAILS 5.6 — case-insensitive unit lookup.
    Falls back to "default" if the unit isn't found.
    """
    usage = citizen.get("usage", {})
    unit_lower = unit.lower().strip() if unit else ""
    # Try exact case-insensitive match first
    for key, qty in usage.items():
        if key.lower() == unit_lower:
            return float(qty)
    # Fall back to "default"
    return float(usage.get("default", 1))


def _money_feel(felt_money: float, noticed: bool) -> str:
    """SPEC §10.1 label table for money impact."""
    if not noticed:
        return "didn't notice"
    if felt_money < 0:
        return "a nice surprise"           # price cut
    if felt_money < 0.3:
        return "noticed, doesn't matter"
    if felt_money < 1.0:
        return "noticeable"
    if felt_money < 2.0:
        return "stings"
    return "hurts"


def _fairness_feel(fairness_pain: float) -> str:
    """SPEC §10.1 label table for fairness."""
    if fairness_pain < 0.5:
        return "seems fair"
    if fairness_pain < 1.5:
        return "feels a bit much"
    if fairness_pain < 3.0:
        return "feels unfair"
    return "feels insulting"


def _change_feel(change_tolerance: float) -> str:
    """SPEC §10.1 label for status-quo bias (used in every decision)."""
    if change_tolerance < 0.35:
        return "You generally resist changes to how things work."
    if change_tolerance < 0.60:
        return "You can live with change if it's explained."
    return "You're usually open to change."


def _is_valid_price(price) -> bool:
    """GUARDRAILS 5.1 — price must be a number in (0, 100 000]."""
    try:
        p = float(price)
        return 0 < p <= 100_000
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# §10.1  perceive — how a decision lands for one citizen
# ---------------------------------------------------------------------------

def perceive(citizen: dict, decision: dict, trust: float) -> dict:
    """
    Compute how a decision lands for one citizen.
    Pure Python — Gemma receives the summary and must not contradict it.

    Returns a dict with all intermediate values (for display and tests)
    plus a plain-English 'summary' (from SPEC §10.1 template rules).

    For non-price decisions (null or invalid prices):
      Returns a simplified dict with only change_feel and summary.
    """
    old_price = decision.get("old_price")
    new_price = decision.get("new_price")
    unit = decision.get("unit", "default")
    decision_maker = decision.get("decision_maker") or "the decision-maker"

    change_tolerance = citizen.get("change_tolerance", 0.5)
    change_feel_text = _change_feel(change_tolerance)

    # --- GUARDRAILS 5.1: price sanity; 5.4: missing old price ---------------
    is_price_decision = _is_valid_price(old_price) and _is_valid_price(new_price)

    if not is_price_decision:
        # Non-price decision — no money math
        summary = (
            "This isn't about money. "
            + change_feel_text
            + " Judge it by your values, constraints and stakes."
        )
        if trust < 4:
            summary += f" You don't fully trust {decision_maker} right now."
        return {
            "is_price": False,
            "change_feel": change_feel_text,
            "summary": summary,
        }

    old_price = float(old_price)
    new_price = float(new_price)

    # --- GUARDRAILS 5.3: equal prices ----------------------------------------
    if old_price == new_price:
        summary = "The price hasn't changed. " + change_feel_text
        return {
            "is_price": True,
            "price_change_pct": 0.0,
            "noticed": False,
            "extra_spend": 0.0,
            "budget_bite_pct": 0.0,
            "money_feel": "didn't notice",
            "fairness_feel": "seems fair",
            "felt_money": 0.0,
            "fairness_pain": 0.0,
            "baseline_qty": _get_baseline_qty(citizen, unit),
            "change_feel": change_feel_text,
            "summary": summary,
        }

    # --- SPEC §10.1 perception math ------------------------------------------

    delta_p = new_price - old_price
    # GUARDRAILS 5.2 — old_price == 0 already excluded by _is_valid_price
    pct = delta_p / old_price

    # Weber's law: the just-noticeable difference scales with trust.
    # Low trust → suspicious → notices smaller changes.
    notice_eff = citizen["notice_threshold"] * (0.5 + trust / 10)
    noticed = abs(pct) >= notice_eff

    baseline_qty = _get_baseline_qty(citizen, unit)
    extra_spend = delta_p * baseline_qty
    # GUARDRAILS 5.2 — monthly_budget == 0 guard
    monthly_budget = citizen.get("monthly_budget", 1) or 1
    bite = extra_spend / monthly_budget

    # GUARDRAILS 5.2 — pain_threshold == 0 guard
    pain_threshold = citizen.get("pain_threshold", 0.01) or 0.01
    money_pain = bite / pain_threshold

    # Kahneman & Tversky loss aversion: a loss feels worse than an equal gain.
    if delta_p > 0:
        felt_money = money_pain * citizen.get("loss_aversion", 1.0)
    else:
        felt_money = money_pain   # cuts are felt less (no aversion multiplier)
    if not noticed:
        felt_money = 0.0          # can't feel what you didn't register

    # Kahneman–Knetsch–Thaler reference price & fairness.
    # price_anchor < 1 → "things used to be cheaper" → higher fairness pain.
    reference = old_price * citizen.get("price_anchor", 1.0)
    # GUARDRAILS 5.2 — reference == 0 guard
    if reference == 0:
        fairness_pain = 0.0
    else:
        ref_gap = (new_price - reference) / reference
        # (5 − trust): lower trust amplifies the sense of unfairness.
        fairness_pain = (
            citizen.get("fairness_weight", 0.5)
            * max(0.0, ref_gap)
            * 10
            * (1 + (5 - trust) / 10)
        )

    money_feel_text = _money_feel(felt_money, noticed)
    fairness_feel_text = _fairness_feel(fairness_pain)

    # --- Build plain-English summary (SPEC §10.1 template) ------------------
    if not noticed:
        money_sentence = "You didn't notice the change."
    elif delta_p < 0:
        money_sentence = (
            f"You noticed. Rs{abs(extra_spend):.0f} back is"
            f" {abs(bite) * 100:.1f}% of your month — {money_feel_text}."
        )
    else:
        money_sentence = (
            f"You noticed. Rs{extra_spend:.0f} extra is"
            f" {bite * 100:.1f}% of your month, and that {money_feel_text}."
        )

    fairness_sentence = f"It {fairness_feel_text}."
    summary = f"{money_sentence} {fairness_sentence} {change_feel_text}"

    if trust < 4:
        summary += f" You don't fully trust {decision_maker} right now."

    return {
        "is_price": True,
        "price_change_pct": round(pct * 100, 2),
        "noticed": noticed,
        "extra_spend": round(extra_spend, 2),
        "budget_bite_pct": round(bite * 100, 2),
        "felt_money": round(felt_money, 4),
        "fairness_pain": round(fairness_pain, 4),
        "money_feel": money_feel_text,
        "fairness_feel": fairness_feel_text,
        "baseline_qty": baseline_qty,
        "change_feel": change_feel_text,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# §10.2  new_quantity — map action + reduce_level → new quantity
# ---------------------------------------------------------------------------

def new_quantity(citizen: dict, action: str, reduce_level: str | None,
                 unit: str = "default") -> float:
    """
    SPEC §10.2 — return the new quantity this citizen will buy.
    """
    baseline = _get_baseline_qty(citizen, unit)
    multipliers = {
        "keep":      1.0,
        "increase":  1.2,
        "switch":    0.0,
        "drop_out":  0.0,
    }
    reduce_multipliers = {
        "light": 0.75,
        "half":  0.50,
        "most":  0.20,
    }
    if action == "reduce":
        mult = reduce_multipliers.get(reduce_level or "half", 0.50)
    else:
        mult = multipliers.get(action, 1.0)

    return round(baseline * mult, 4)


# ---------------------------------------------------------------------------
# §10.3  compute_metrics — per-drive aggregate numbers
# ---------------------------------------------------------------------------

def compute_metrics(
    citizens: list[dict],
    reactions: list[dict],
    perceptions: list[dict],
    decision: dict,
    trusts_before: list[float],
) -> dict:
    """
    SPEC §10.3 — compute all aggregate metrics for one drive.

    citizens:     list of 6 citizen dicts (from crowds.json)
    reactions:    list of 6 reaction dicts from Gemma (may include fallbacks)
    perceptions:  list of 6 perception dicts from perceive()
    decision:     the decision dict
    trusts_before: list of 6 trust values before this drive
    """
    old_price = decision.get("old_price")
    new_price = decision.get("new_price")
    unit = decision.get("unit", "default")
    is_price = _is_valid_price(old_price) and _is_valid_price(new_price)

    if is_price:
        old_price = float(old_price)
        new_price = float(new_price)

    # --- quantities and revenue ----------------------------------------------
    old_quantities = []
    new_quantities = []
    for citizen, reaction, perception in zip(citizens, reactions, perceptions):
        baseline = perception.get("baseline_qty", _get_baseline_qty(citizen, unit))
        old_quantities.append(baseline)
        action = reaction.get("action", "keep")
        reduce_level = reaction.get("reduce_level")
        new_quantities.append(new_quantity(citizen, action, reduce_level, unit))

    Q0 = sum(old_quantities)
    Q1 = sum(new_quantities)

    if is_price:
        old_revenue = round(Q0 * old_price, 2)
        new_revenue = round(Q1 * new_price, 2)
        # GUARDRAILS 5.2 — avoid div/0 if old_revenue is 0
        if old_revenue != 0:
            revenue_change_pct = round((new_revenue - old_revenue) / old_revenue * 100, 1)
        else:
            revenue_change_pct = None

        # Arc elasticity (SPEC §10.3) — symmetric mid-point formula
        # GUARDRAILS 5.2 — guard against Q0+Q1==0 and P0+P1==0
        avg_q = (Q0 + Q1) / 2
        avg_p = (old_price + new_price) / 2
        if avg_q != 0 and avg_p != 0 and old_price != new_price:
            pct_delta_q = (Q1 - Q0) / avg_q
            pct_delta_p = (new_price - old_price) / avg_p
            elasticity = round(pct_delta_q / pct_delta_p, 2)
            elasticity_label = "elastic" if abs(elasticity) > 1 else "inelastic"
        else:
            elasticity = None
            elasticity_label = "N/A"
    else:
        old_revenue = new_revenue = revenue_change_pct = None
        elasticity = None
        elasticity_label = "N/A"

    # --- Hirschman: Exit / Voice / Loyalty / Grudging -----------------------
    exits = []
    voices = []
    loyals = []
    grudges = []
    say_do_gaps = 0

    for reaction in reactions:
        action = reaction.get("action", "keep")
        voice = reaction.get("voice", "silent")
        sentiment = int(reaction.get("sentiment", 0))
        is_exit = action in ("switch", "drop_out")

        if is_exit:
            exits.append(reaction)
        else:
            if voice in ("complain_publicly", "rally_others"):
                voices.append(reaction)          # voice only from non-exiters
            if action in ("keep", "increase") and sentiment >= 0:
                loyals.append(reaction)
            elif action in ("keep", "reduce") and sentiment < 0:
                grudges.append(reaction)

        # Say–do gap: defending while reducing/exiting, OR
        # positive sentiment while exiting
        if (voice == "defend" and action in ("reduce", "switch", "drop_out")) or \
           (sentiment >= 0 and is_exit):
            say_do_gaps += 1

    # --- averages and hurt-most ----------------------------------------------
    sentiments = [int(r.get("sentiment", 0)) for r in reactions]
    avg_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

    avg_trust = round(sum(trusts_before) / len(trusts_before), 2) if trusts_before else 0

    # Hurt-most: lowest sentiment; tie → highest felt_money
    hurt_most_id = None
    if reactions:
        worst_sentiment = min(sentiments)
        candidates = [
            (reactions[i], perceptions[i])
            for i in range(len(reactions))
            if int(reactions[i].get("sentiment", 0)) == worst_sentiment
        ]
        if len(candidates) == 1:
            hurt_most_id = candidates[0][0].get("citizen_id")
        else:
            # Tiebreak: highest felt_money
            hurt_most_id = max(
                candidates,
                key=lambda pair: pair[1].get("felt_money", 0)
            )[0].get("citizen_id")

    return {
        "old_revenue": old_revenue,
        "new_revenue": new_revenue,
        "revenue_change_pct": revenue_change_pct,
        "elasticity": elasticity,
        "elasticity_label": elasticity_label,
        "Q0": round(Q0, 2),
        "Q1": round(Q1, 2),
        "exit_count": len(exits),
        "voice_count": len(voices),
        "loyalty_count": len(loyals),
        "grudging_count": len(grudges),
        "say_do_gaps": say_do_gaps,
        "avg_sentiment": avg_sentiment,
        "avg_trust": avg_trust,
        "hurt_most": hurt_most_id,
    }


# ---------------------------------------------------------------------------
# §10.4  update_trust
# ---------------------------------------------------------------------------

def update_trust(trust: float, sentiment: int, action: str, voice: str) -> float:
    """
    SPEC §10.4 — update trust after a reaction.
    Rounded to 1 decimal, clamped to [0, 10].

    The 0.5 multiplier: trust moves slower than sentiment — one bad
    experience doesn't destroy a relationship, but exit nearly does.
    """
    is_exit = action in ("switch", "drop_out")
    is_defend = voice == "defend"
    new_trust = (
        trust
        + 0.5 * sentiment                   # sentiment moves trust (halved)
        - (0.7 if is_exit else 0.0)         # exit: strong signal of lost trust
        + (0.2 if is_defend else 0.0)       # defend: small loyalty bonus
    )
    return round(clamp(new_trust, 0.0, 10.0), 1)
