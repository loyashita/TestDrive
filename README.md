# Test Drive 🚗

**Test-drive any decision on a crowd that remembers.**

Snap a photo of a poster, menu or notice (or type a decision). Six written characters live through it: what they *say*, what they secretly *think*, what they actually *do*. A director gives notes and a tweak. Drive the tweak, and the crowd remembers what you did last time.

Built for Hacktoberfest Hack Day Zuarinagar x SEDS Celestia (BITS Goa). Tracks: Best Use of Gemma 4, Best Open-Source AI Project.

<!-- TODO: add a screenshot of one drive on Ollama here: ![Test Drive](docs/screenshot.png) -->

## The idea: Gemma decides, Python counts

| Python (`engine.py`, no model calls) | Gemma (`model.py`) |
|---|---|
| Did they notice the change? How much does it bite their budget? Does it feel fair? | What they say out loud |
| Quantities, revenue, elasticity | What they really think |
| Exit / voice / loyalty tally, trust updates | What they do, and why |

Perception is computed in Python and *handed* to the model as fact, so the model can't hand-wave the money.

## Honesty principle

- **Not a forecast.** The characters are fictional, written by us and played by a language model. They surface perspectives you missed. They don't replace asking people; they tell you who to ask and what to ask.
- **"Learning" is in-context memory, not training.** After each drive, notes, trust and outlook are saved to `memory/*.json` and the last 3 memories are pasted into the next prompt. The model's weights never change.
- **The maths is a model, not truth.** Thresholds are our assumptions.
- All characters are fictional; any resemblance to real BITS Goa students is coincidental.

## Behavioural-economics grounding

- **Weber's law**: the just-noticeable difference (`notice_threshold`), loosened or tightened by trust.
- **Loss aversion** (Kahneman & Tversky): price rises are felt `loss_aversion`× more than equal cuts.
- **Reference prices and fairness** (Kahneman, Knetsch & Thaler): each person has a "fair price" (`price_anchor`); you can be insulted by a price you can easily afford.
- **Status-quo bias**: `change_tolerance` shapes how any change lands, price or not.
- **Exit, Voice and Loyalty** (Hirschman): every drive tallies who left, who complained, who stayed loyal and who stayed grudgingly.
- **Arc elasticity** for the demand response.

Formulas live in `engine.py`, each with a comment naming the idea.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your values
streamlit run app.py
```

**Gemini API (default, faster):** put your key in `.env` as `GEMINI_API_KEY` and set `GEMINI_MODEL` to a Gemma 4 model ID from [AI Studio](https://aistudio.google.com/).

**Ollama (local, offline):**
```bash
ollama pull gemma4:e4b
# in .env:  MODEL_BACKEND=ollama
```
You can also switch backend from the app sidebar.

## Terminal mode

```bash
python rehearsal.py            # three drives: fest pass 349→499, 499→549, a non-price one
python rehearsal.py reset      # wipe memories first
python rehearsal.py ollama 2   # only drive 2, on Ollama
```

## Tests (no model or network needed)

```bash
python test_engine.py      # engine vs. the hand-calculated case
python test_rehearsal.py   # memory, validation, fallbacks, persistence
```

## Safety notes

Decisions are screened before running; user text and photo text are treated as data, not instructions; every model output is validated (allowed values, length caps) with retry and a flagged fallback; the API key lives only in `.env`. Known limitation: two browser tabs writing at once means the last write wins.

## Layout

```
app.py        Streamlit UI            memory.py   load/save/reset state
rehearsal.py  orchestration           prompts.py  all prompts
engine.py     perception + maths      model.py    Gemini/Ollama client, retry
crowds.json   the cast                memory/     runtime state (gitignored)
```

## License

MIT
