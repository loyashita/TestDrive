"""
app.py — Streamlit UI for Test Drive.  Run: streamlit run app.py

Key concept: Streamlit reruns this whole script top to bottom on every click.
  - st.session_state survives reruns but dies when you close the tab
    (it holds the drives shown on screen, the "running" flag, the pending job).
  - memory/*.json on disk survives everything (it holds trust and memories).

GUARDRAILS covered here: 2.3/2.4 (input caps), 3.4/3.5 (honesty note, AI label),
5.4 (missing old price), 6.5 (reset confirmation), 7.1 (button locked while
running), 7.2 (progress), 7.3 (friendly errors), 7.4 (cached crowds), 7.5 (secrets hidden).
"""

import pandas as pd
import streamlit as st

import memory
import model
import rehearsal

st.set_page_config(page_title="Test Drive", page_icon="🚗", layout="wide")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SENTIMENT_EMOJI = {-3: "😡", -2: "😠", -1: "🙁", 0: "😐", 1: "🙂", 2: "😊", 3: "🤩"}
FEEL_COLOR = {"didn't notice": "gray", "a nice surprise": "green", "noticed, doesn't matter": "gray",
              "noticeable": "orange", "stings": "orange", "hurts": "red"}
FAIR_COLOR = {"seems fair": "green", "feels a bit much": "orange",
              "feels unfair": "red", "feels insulting": "red"}


@st.cache_data
def get_crowds():
    return memory.load_crowds()


ss = st.session_state
ss.setdefault("drives", [])      # drive results shown on screen
ss.setdefault("running", False)  # GUARDRAILS 7.1
ss.setdefault("job", None)       # what to run on the next rerun
ss.setdefault("messages", [])    # (kind, text) shown after a run


# --- sidebar -----------------------------------------------------------------

crowds = get_crowds()
with st.sidebar:
    st.title("🚗 Test Drive")
    crowd = st.selectbox("Crowd", crowds, format_func=lambda c: c["title"])
    backend = st.selectbox("Backend", ["gemini", "ollama"],
                           index=["gemini", "ollama"].index(model.MODEL_BACKEND)
                           if model.MODEL_BACKEND in ("gemini", "ollama") else 0)
    model.MODEL_BACKEND = backend

    state, mem_warning = memory.load_state(crowd)
    st.metric("Drives so far", memory.next_drive_number(state) - 1)
    reveal = st.toggle("Reveal secrets", value=False)

    confirm = st.checkbox("Yes, wipe memories")
    if st.button("Reset memories", disabled=not confirm or ss.running):
        memory.reset_state(crowd)
        ss.drives = []
        st.rerun()

    st.info(
        "Characters are fictional (any resemblance to real people is coincidental) and are "
        "played by an AI model. This is not a forecast of what real people will do: it "
        "doesn't replace asking them, it tells you who to ask. Memory is saved notes pasted "
        "into each prompt, not training."
    )
    st.caption("Characters played by Gemma 4")

if mem_warning:
    st.warning(mem_warning)


# --- input -------------------------------------------------------------------

def start_job(job):
    ss.job = job
    ss.running = True


st.header("Test-drive a decision")
c_in, c_price = st.columns([3, 1])
with c_in:
    text = st.text_area("Describe it (or upload a photo of the poster/menu/notice)",
                        max_chars=rehearsal.MAX_TEXT_CHARS, height=80,
                        placeholder="e.g. Library closes at 11pm instead of 2am")
    upload = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "webp"])
with c_price:
    st.caption("Price override (0 = leave as read)")
    old_override = st.number_input("Old price (Rs)", min_value=0.0, step=10.0)
    new_override = st.number_input("New price (Rs)", min_value=0.0, step=10.0)

image = None
if upload is not None:
    if upload.size > MAX_UPLOAD_BYTES:
        st.error("That image is over 5 MB. Please use a smaller one.")
    else:
        image = upload.getvalue()
        st.image(image, width=240)

st.button("Test drive", type="primary", disabled=ss.running,
          on_click=start_job,
          args=({"kind": "new", "text": text, "image": image,
                 "old": old_override, "new": new_override},))


# --- running a job -----------------------------------------------------------

def run_job(job):
    """Turn the job into a drive result. Raises nothing the user should see raw."""
    if job["kind"] == "tweak":
        decision = job["decision"]
    else:
        decision, err = rehearsal.extract_decision(job["image"], job["text"])
        if err:
            ss.messages.append(("error", err))
            return
        if not decision["safe"]:   # GUARDRAILS 3.1
            ss.messages.append(("info", decision["safety_reason"]))
            return
        if job["old"] > 0:
            decision["old_price"] = job["old"]
        if job["new"] > 0:
            decision["new_price"] = job["new"]
        if decision["new_price"] and not decision["old_price"]:   # GUARDRAILS 5.4
            ss.messages.append(("warning", "I only found one price, so this is treated as a non-price "
                                           "decision. Fill in the old price to run the money maths."))

    bar = st.progress(0.0)
    status = st.empty()

    def progress(i, n, name):
        bar.progress((i - 1) / n)
        status.write(f"{name} is thinking… ({i}/{n})")   # GUARDRAILS 7.2

    result = rehearsal.run_drive(crowd, decision, progress=progress)
    status.write("Director is writing notes…")
    if result["unreliable"]:
        ss.messages.append(("error", f"{result['fallbacks']} of 6 reactions failed, so this drive was "
                                     "not saved. Retry, or switch backend."))
    else:
        result["notes"] = rehearsal.direct(result)
        result["headline"] = rehearsal.headline(result)
    ss.drives.append(result)
    bar.empty()
    status.empty()


if ss.running:
    try:
        run_job(ss.job)
    except Exception as e:   # GUARDRAILS 7.3: never show a raw traceback
        ss.messages.append(("error", f"Something went wrong ({type(e).__name__}). "
                                     "Check the backend and try again."))
    ss.running = False
    ss.job = None
    st.rerun()

for kind, msg in ss.messages:
    {"error": st.error, "info": st.info}.get(kind, st.warning)(msg)
ss.messages = []


# --- rendering a drive -------------------------------------------------------

def card(c, p, r, t0, t1, cs):
    with st.container(border=True):
        st.subheader(f"{c['name'].split()[0]}, {c['age']}")
        st.caption(c["role"])
        if p["is_price"]:
            st.markdown(f":{FEEL_COLOR[p['money_feel']]}-badge[{p['money_feel']}] "
                        f":{FAIR_COLOR[p['fairness_feel']]}-badge[{p['fairness_feel']}]")
        else:
            st.markdown(":gray-badge[not about money]")
        if "_warning" in r:
            st.warning("⚠ The model failed for this person; nothing was recorded.")
        else:
            level = f" ({r['reduce_level']})" if r["reduce_level"] else ""
            st.markdown(f":blue-badge[{r['action']}{level}] :violet-badge[{r['voice']}] "
                        f"{SENTIMENT_EMOJI[r['sentiment']]}")
            st.markdown(f"💬 {r['quote']}")
            st.markdown(f"<span style='color:gray;font-style:italic'>💭 {r['inner_thought']}</span>",
                        unsafe_allow_html=True)
        arrow = "▲" if t1 > t0 else "▼" if t1 < t0 else "–"
        st.markdown(f"**Trust** {t0:.1f} → {t1:.1f} {arrow}")
        st.caption(f"Outlook: {cs['outlook']}")
        with st.expander("remembers…"):
            st.text(memory.format_memories(cs["memories"]))
        if reveal:
            st.markdown(f"🔒 **Secret:** {c['secret']}")


def render_drive(res):
    d = res["decision"]
    st.subheader(f"Drive {res['drive_number']}: {rehearsal.decision_label(d)}")
    if res.get("headline"):
        st.markdown(f"### 📰 {res['headline']}")
    if res["unreliable"]:
        st.error("Unreliable drive: not saved.")

    state_now, _ = memory.load_state(crowd)
    cols = st.columns(3)
    rows = list(zip(res["citizens"], res["perceptions"], res["reactions"],
                    res["trusts_before"], res["trusts_after"]))
    for i, (c, p, r, t0, t1) in enumerate(rows):
        with cols[i % 3]:
            card(c, p, r, t0, t1, state_now[c["id"]])

    m = res["metrics"]
    k = st.columns(5)
    k[0].metric("Revenue", "n/a" if m["revenue_change_pct"] is None else f"{m['revenue_change_pct']:+.1f}%")
    k[1].metric("Elasticity", "N/A" if m["elasticity"] is None else f"{m['elasticity']} ({m['elasticity_label']})")
    k[2].metric("Exit / Voice", f"{m['exit_count']} / {m['voice_count']}")
    k[3].metric("Loyal / Grudging", f"{m['loyalty_count']} / {m['grudging_count']}")
    k[4].metric("Avg trust", f"{m['avg_trust_after']:.1f}", f"{m['avg_trust_after'] - m['avg_trust']:+.1f}")
    st.caption(f"Say–do gaps: {m['say_do_gaps']}")

    notes = res.get("notes")
    if notes:
        names = {c["id"]: c["name"] for c in res["citizens"]}
        with st.container(border=True):
            st.markdown("#### 🎬 Director's notes")
            st.markdown(f"**Hurt most:** {names.get(notes['hurt_most'], notes['hurt_most'])}")
            st.markdown(f"**Missed objection:** {notes['missed_objection']}")
            st.markdown(f"**Hidden issue:** {notes['hidden_issue']}")
            st.markdown(f"**Trust warning:** {notes['trust_warning']}")
            st.markdown(f"**Tweak:** {notes['tweak']}")
            t = notes["tweaked_decision"]
            st.button("Test-drive the tweak", key=f"tweak{res['drive_number']}_{len(ss.drives)}",
                      disabled=ss.running, on_click=start_job,
                      args=({"kind": "tweak", "decision": t},))
    elif not res["unreliable"]:
        st.info("The director had no notes this time.")


# --- output ------------------------------------------------------------------

if ss.drives:
    render_drive(ss.drives[-1])   # newest on top; older drives (B under A) below
    for older in reversed(ss.drives[:-1]):
        with st.expander(f"Earlier: Drive {older['drive_number']} — {rehearsal.decision_label(older['decision'])}"):
            render_drive(older)

    st.subheader("Trust over time")
    hist = {c["name"].split()[0]: pd.Series(state[c["id"]]["trust_history"], dtype=float)
            for c in crowd["citizens"]}
    st.line_chart(pd.DataFrame(hist))
else:
    st.caption("Add a decision above and press Test drive.")
