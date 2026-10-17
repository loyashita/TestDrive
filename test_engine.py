"""
test_engine.py — verify engine.py against the SPEC §10.5 hand test case.

BUILD_PLAN: "Before running, Yashita solves §10.5 on paper.
Done when the code equals the paper."

Run with: python test_engine.py
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine import perceive, new_quantity, compute_metrics, update_trust


# ---------------------------------------------------------------------------
# Load the cast
# ---------------------------------------------------------------------------

with open("crowds.json", encoding="utf-8") as f:
    data = json.load(f)

crowd = data["crowds"][0]
citizens = {c["id"]: c for c in crowd["citizens"]}

# Convenience: fetch citizen by name initial
A = citizens["c1"]  # Ananya
S = citizens["c2"]  # Siddharth
M = citizens["c3"]  # Meghna
R = citizens["c4"]  # Rohit
Sa = citizens["c5"] # Sana
T = citizens["c6"]  # Tenzin

all_citizens = [A, S, M, R, Sa, T]
starting_trusts = [c["starting_trust"] for c in all_citizens]

DECISION = {
    "title": "Fest pass price increase",
    "description": "Fest pass goes from Rs349 to Rs499",
    "old_price": 349,
    "new_price": 499,
    "unit": "fest pass",
    "decision_maker": "the fest organising committee",
}

REACTIONS = [
    {"citizen_id": "c1", "action": "reduce",   "reduce_level": "most",  "voice": "silent",       "sentiment": -2},
    {"citizen_id": "c2", "action": "drop_out", "reduce_level": None,    "voice": "rally_others", "sentiment": -3},
    {"citizen_id": "c3", "action": "keep",     "reduce_level": None,    "voice": "defend",       "sentiment":  1},
    {"citizen_id": "c4", "action": "reduce",   "reduce_level": "light", "voice": "silent",       "sentiment": -1},
    {"citizen_id": "c5", "action": "drop_out", "reduce_level": None,    "voice": "silent",       "sentiment": -1},
    {"citizen_id": "c6", "action": "keep",     "reduce_level": None,    "voice": "silent",       "sentiment":  0},
]


# ---------------------------------------------------------------------------
# §10.1 Perception — Ananya, Siddharth, Sana
# ---------------------------------------------------------------------------

print("=" * 60)
print("SPEC §10.1 — Perception")
print("=" * 60)

perceptions = []
for citizen, reaction in zip(all_citizens, REACTIONS):
    p = perceive(citizen, DECISION, citizen["starting_trust"])
    perceptions.append(p)

for cit, p in zip(all_citizens, perceptions):
    if cit["id"] in ("c1", "c2", "c5"):
        print(f"\n{cit['name']} ({cit['id']})")
        print(f"  noticed:      {p['noticed']}")
        print(f"  bite %:       {p['budget_bite_pct']:.2f}%")
        print(f"  money_feel:   {p['money_feel']}")
        print(f"  fairness_feel:{p['fairness_feel']}")
        print(f"  summary:      {p['summary']}")

# Hardcoded expected values from the hand calculation (see SPEC §10.5)
print("\n--- Checking against expected values ---")

def check(label, got, expected, tol=0.05):
    ok = abs(got - expected) <= tol if isinstance(got, float) else got == expected
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}: got={got!r}, expected={expected!r}")
    return ok

all_ok = True

# Ananya: noticed (42.98% > 4.4%), bite 6.52%, money hurts, fairness a bit much
p_a = perceptions[0]
all_ok &= check("Ananya noticed",       p_a["noticed"],          True)
all_ok &= check("Ananya bite %",        p_a["budget_bite_pct"],  6.52, tol=0.1)
all_ok &= check("Ananya money_feel",    p_a["money_feel"],       "hurts")
all_ok &= check("Ananya fairness_feel", p_a["fairness_feel"],    "feels a bit much")

# Siddharth: NOT noticed (42.98% < 48%), money "didn't notice", fairness "feels insulting"
p_s = perceptions[1]
all_ok &= check("Siddharth noticed",       p_s["noticed"],          False)
all_ok &= check("Siddharth money_feel",    p_s["money_feel"],       "didn't notice")
all_ok &= check("Siddharth fairness_feel", p_s["fairness_feel"],    "feels insulting")

# Sana: noticed, bite 4.17%, stings, feels unfair
p_sa = perceptions[4]
all_ok &= check("Sana noticed",       p_sa["noticed"],          True)
all_ok &= check("Sana bite %",        p_sa["budget_bite_pct"],  4.17, tol=0.1)
all_ok &= check("Sana money_feel",    p_sa["money_feel"],       "stings")
all_ok &= check("Sana fairness_feel", p_sa["fairness_feel"],    "feels unfair")


# ---------------------------------------------------------------------------
# §10.3 Metrics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SPEC §10.3 — Metrics")
print("=" * 60)

metrics = compute_metrics(all_citizens, REACTIONS, perceptions, DECISION, starting_trusts)

print(f"\n  Old revenue:       Rs{metrics['old_revenue']:.2f}  (expected: Rs3490.00)")
print(f"  New revenue:       Rs{metrics['new_revenue']:.2f}")
print(f"  Revenue change %:  {metrics['revenue_change_pct']:.1f}%")
print(f"  Q0 (old qty):      {metrics['Q0']}  (expected: 10)")
print(f"  Q1 (new qty):      {metrics['Q1']}  (expected: 5.95)")
print(f"  Elasticity:        {metrics['elasticity']}  (expected: ~-1.44, elastic)")
print(f"  Elasticity label:  {metrics['elasticity_label']}")
print(f"  Exit:              {metrics['exit_count']}  (Siddharth + Sana = 2)")
print(f"  Voice:             {metrics['voice_count']}  (non-exiters who complain/rally = 0)")
print(f"  Loyalty:           {metrics['loyalty_count']}  (Meghna +1, Tenzin 0 = 2)")
print(f"  Grudging:          {metrics['grudging_count']}  (Ananya -2, Rohit -1 = 2)")
print(f"  Say-do gaps:       {metrics['say_do_gaps']}")
print(f"  Hurt-most:         {metrics['hurt_most']}  (expected: c2 Siddharth, sentiment -3)")

all_ok &= check("Old revenue",    metrics["old_revenue"],         3490.0,  tol=0.5)
all_ok &= check("Q0",             metrics["Q0"],                  10.0,    tol=0.01)
all_ok &= check("Q1",             metrics["Q1"],                  5.95,    tol=0.01)
all_ok &= check("Elasticity",     metrics["elasticity"],          -1.44,   tol=0.05)
all_ok &= check("Exit count",     float(metrics["exit_count"]),   2.0)
all_ok &= check("Voice count",    float(metrics["voice_count"]),  0.0)
all_ok &= check("Loyalty count",  float(metrics["loyalty_count"]),2.0)
all_ok &= check("Grudging count", float(metrics["grudging_count"]),2.0)
all_ok &= check("Hurt-most",      metrics["hurt_most"],           "c2")


# ---------------------------------------------------------------------------
# §10.4 Trust update
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SPEC §10.4 — Trust update")
print("=" * 60)

trust_results = []
for citizen, reaction in zip(all_citizens, REACTIONS):
    t_before = citizen["starting_trust"]
    t_after = update_trust(
        t_before,
        reaction["sentiment"],
        reaction["action"],
        reaction["voice"],
    )
    trust_results.append((citizen["name"], t_before, t_after))
    print(f"  {citizen['name']:20s}  {t_before} -> {t_after}")

# Expected: Ananya 5.0, Siddharth 0.8, Meghna 7.7, Rohit 4.5, Sana 2.8, Tenzin 4.0
expected_trusts = {"c1": 5.0, "c2": 0.8, "c3": 7.7, "c4": 4.5, "c5": 2.8, "c6": 4.0}
for i, (cit, reaction) in enumerate(zip(all_citizens, REACTIONS)):
    t_after = trust_results[i][2]
    all_ok &= check(f"Trust {cit['name']}", t_after, expected_trusts[cit["id"]], tol=0.05)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — engine matches the hand calculation.")
else:
    print("SOME CHECKS FAILED — review the values above.")
    sys.exit(1)
