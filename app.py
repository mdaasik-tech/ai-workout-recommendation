"""Plate - AI Workout Recommendation System (Streamlit)."""
from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import config as C
from src.data_loader import DataValidationError, load_exercises
from src.ml_ranker import (CAT, ContentSimilarity, append_feedback, load_active_model, load_feedback,
                           load_report, train_feedback_model)
from src.ui import CSS, barbell, esc, h, stagger_css, stats_component
from src.validators import UserProfile, validate_profile
from src.workout_builder import (build_workout, get_alternatives, replace_exercise, workout_to_csv, workout_to_text)

st.set_page_config(page_title="Plate - AI Workout Recommendation", page_icon="🏋️", initial_sidebar_state="collapsed")
st.markdown(CSS + stagger_css(), unsafe_allow_html=True)

PAGES = {"home": "Home", "create": "Create Workout", "workout": "My Workout", "ml": "ML Lab", "about": "About & Safety"}
GYM_EQ = ["barbell", "dumbbells", "cable_machine", "machines", "pull_up_bar"]
HOME_EQ = ["dumbbells", "resistance_bands", "bodyweight"]
STEP_NAMES = ["You & goal", "Experience", "Muscle group", "Gear & time", "Limits"]
pretty = lambda s: str(s).replace("_", " ").capitalize()


# ------------------------------------------------------------------ data & state
@st.cache_data
def get_df() -> pd.DataFrame:
    return load_exercises()


@st.cache_resource
def get_sim() -> ContentSimilarity:
    return ContentSimilarity(get_df())


@st.cache_resource
def _load_model(stamp: float):
    return load_active_model()


def active_model():
    stamp = C.MODEL_REPORT_FILE.stat().st_mtime if C.MODEL_REPORT_FILE.exists() else 0.0
    return _load_model(stamp)


try:
    DF = get_df()
except (DataValidationError, FileNotFoundError) as e:
    st.error(f"Could not load exercises.csv.\n\n{e}")
    st.stop()
SIM = get_sim()

ss = st.session_state
ss.setdefault("page", "home")
ss.setdefault("step", 1)
ss.setdefault("ans", dict(name="", age=None, goal="", level="", muscle="", emphasis=[], loc="gym", eq=list(GYM_EQ),
                          dur=45, intensity="moderate", pain=False, inj=False, med=False, areas=[], avoid="", prefer=""))
ss.setdefault("workout", None)
ss.setdefault("profile", None)
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("use_ml", True)
ss.setdefault("flash", None)
ss.setdefault("fb_sent", False)
A = ss.ans


def goto(page: str):
    ss.page = page


def set_ans(key, value):
    A[key] = value


def profile_from_answers() -> UserProfile:
    return UserProfile(goal=A["goal"], level=A["level"], muscle_group=A["muscle"], emphasis=A["emphasis"], location=A["loc"],
                       equipment=A["eq"], duration=A["dur"], intensity=A["intensity"], pain=A["pain"], injury=A["inj"],
                       medical=A["med"], protect_areas=A["areas"], avoid_text=A["avoid"], prefer_text=A["prefer"],
                       name=A["name"], age=A["age"])


def model_in_use():
    return active_model() if ss.use_ml else None


# ------------------------------------------------------------------ chrome
def header():
    c = st.columns([1.1, 4])
    c[0].markdown('<div class="logo"><i></i>Plate</div>', unsafe_allow_html=True)
    nav = c[1].columns(len(PAGES))
    for col, (k, label) in zip(nav, PAGES.items()):
        col.button(label, key=f"nav_{k}", type="primary" if ss.page == k else "secondary", on_click=goto, args=(k,), use_container_width=True)
    st.write("")


def option_cards(key: str, options: list[tuple[str, str, str]], cols: int):
    """Selectable cards built from buttons. The selected card is the primary button."""
    with st.container(key=f"cards_{key}"):
        grid = st.columns(cols)
        for i, (val, title, desc) in enumerate(options):
            grid[i % cols].button(f"**{title}**\n\n{desc}", key=f"card_{key}_{val}", use_container_width=True,
                                  type="primary" if A[key] == val else "secondary", on_click=set_ans, args=(key, val))


# ------------------------------------------------------------------ pages
def page_home():
    left, right = st.columns([1.15, 0.85], vertical_alignment="center")
    with left:
        st.markdown(h("""<div class="hero"><h1>Your next session, planned in seconds.</h1>
            <p>Answer five short questions. Get a structured workout with sets, reps, rest, technique cues and swaps,
            ranked by transparent rules and made smarter with machine learning.</p></div>"""), unsafe_allow_html=True)
        st.button("Create my workout", key="cta", type="primary", on_click=goto, args=("create",))
    with right:
        st.markdown(barbell(all_plates=5, animate_from=0), unsafe_allow_html=True)
    st.markdown(h("""<div class="how"><div><b>Pick your target</b><span>Goal, level and one muscle group.</span></div>
        <div><b>Tell us your gear</b><span>Gym or home, equipment, time, intensity.</span></div>
        <div><b>Train, swap, rate</b><span>Replace any move, download the plan, rate it.</span></div></div>"""), unsafe_allow_html=True)
    counts = DF.muscle_group.value_counts()
    st.markdown("".join(f'<span class="chip">{pretty(m)} · {counts.get(m, 0)}</span>' for m in C.MUSCLE_GROUPS), unsafe_allow_html=True)
    if not DF.reviewed.any():
        st.markdown(f'<div class="warn" style="margin-top:14px"><b>Heads up:</b> all {len(DF)} exercises in the catalogue are marked unreviewed. '
                    'Have a qualified coach review them before sharing this app publicly.</div>', unsafe_allow_html=True)
    st.caption("Educational fitness guidance, not medical advice.")


def step_valid() -> bool:
    return {1: bool(A["goal"]), 2: bool(A["level"]), 3: bool(A["muscle"]), 4: bool(A["eq"]), 5: True}[ss.step]


def page_create():
    n = ss.step
    st.markdown(f'<div class="prog">{barbell(all_plates=n, animate_from=n - 1)}</div><div class="step">Step {n} of 5 · {STEP_NAMES[n - 1]}</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        if n == 1:
            st.subheader("Tell us a little")
            st.markdown('<p class="sub">Name and age are optional. Age is only checked to be in a sensible range.</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            A["name"] = c1.text_input("Display name", value=A["name"], max_chars=30)
            A["age"] = c2.number_input("Age", min_value=13, max_value=90, value=A["age"], step=1, placeholder="Optional")
            st.markdown("**What's your goal?**")
            option_cards("goal", [(k, v[0], v[1]) for k, v in C.GOALS.items()], 2)
        elif n == 2:
            st.subheader("How experienced are you?")
            st.markdown('<p class="sub">Choose by consistency and technique, not just months in a gym.</p>', unsafe_allow_html=True)
            option_cards("level", [(k, pretty(k), v) for k, v in C.LEVEL_INFO.items()], 3)
        elif n == 3:
            st.subheader("Which muscle group today?")
            st.markdown('<p class="sub">One focus per session. Exercises train several muscles, so emphasis is not isolation.</p>', unsafe_allow_html=True)
            option_cards("muscle", [(k, pretty(k), v) for k, v in C.MUSCLE_INFO.items()], 3)
            if A["muscle"]:
                opts = sorted(DF[DF.muscle_group == A["muscle"]].muscle_emphasis.unique())
                A["emphasis"] = st.pills("Emphasis (optional)", opts, selection_mode="multi", format_func=pretty,
                                         default=[e for e in A["emphasis"] if e in opts]) or []
        elif n == 4:
            st.subheader("Gear and time")
            loc = st.pills("Where will you train?", C.LOCATIONS, selection_mode="single", default=A["loc"], format_func=pretty) or A["loc"]
            if loc != A["loc"]:
                A["loc"], A["eq"] = loc, list(GYM_EQ if loc == "gym" else HOME_EQ)
                st.rerun()
            A["eq"] = st.pills("Equipment you have", C.EQUIPMENT, selection_mode="multi", default=A["eq"],
                               format_func=lambda e: C.EQUIPMENT_LABELS[e]) or []
            if not A["eq"]:
                st.caption("Select at least one option. Choose 'Bodyweight' if you have no equipment.")
            A["dur"] = st.pills("Time available", C.DURATIONS, selection_mode="single", default=A["dur"], format_func=lambda d: f"{d} min") or A["dur"]
            A["intensity"] = st.pills("Intensity", C.INTENSITIES, selection_mode="single", default=A["intensity"], format_func=pretty) or A["intensity"]
        else:
            st.subheader("Anything we should know?")
            st.markdown('<p class="sub">We do not diagnose or prescribe rehab. Answers only shape safety messages and exclusions, '
                        'and are never saved or used to train models.</p>', unsafe_allow_html=True)
            A["pain"] = st.toggle("Current pain or movement limitation", value=A["pain"])
            A["inj"] = st.toggle("Past injury relevant to exercise", value=A["inj"])
            A["med"] = st.toggle("Medical restriction affecting exercise", value=A["med"])
            A["areas"] = st.pills("Areas to protect (removes exercises with a caution for that area)", list(C.PROTECT_AREAS),
                                  selection_mode="multi", default=A["areas"], format_func=pretty) or []
            A["avoid"] = st.text_input("Exercises or movements to avoid (comma separated, optional)", value=A["avoid"], placeholder="e.g. squat, lunge")
            A["prefer"] = st.text_input("Exercises you prefer (optional)", value=A["prefer"], placeholder="e.g. row, curl")
    back, nxt = st.columns([1, 1])
    if back.button("Back", key="back"):
        if n == 1:
            ss.page = "home"
        else:
            ss.step -= 1
        st.rerun()
    if n < 5:
        if nxt.button("Continue", key="next", type="primary", disabled=not step_valid(), use_container_width=True):
            ss.step += 1
            st.rerun()
    elif nxt.button("Generate workout", key="gen", type="primary", use_container_width=True):
        generate()


def generate(jitter: float = 0.0, seed: int | None = None, animate: bool = True):
    p = profile_from_answers()
    errors = validate_profile(p)
    if errors:
        for e in errors:
            st.error(e)
        return
    if animate:
        ph = st.empty()
        msgs = ["Matching exercises to your gear…", "Ranking by muscle, level and goal…", "Checking similarity to avoid repeats…", "Fitting the session to your time…"]
        for i in range(6):
            ph.markdown(f'<div class="load">{barbell(all_plates=i, animate_from=max(i - 1, 0))}<h3>{msgs[i % 4]}</h3></div>', unsafe_allow_html=True)
            time.sleep(0.28)
        ph.empty()
    model = model_in_use()
    ss.profile = p
    ss.workout = build_workout(DF, p, SIM, model=model, use_ml=model is not None, jitter=jitter, seed=seed)
    ss.fb_sent, ss.flash, ss.page = False, None, "workout"
    ss.session_id = uuid.uuid4().hex[:12]
    st.rerun()


def swap(idx: int, new_id: str):
    model = model_in_use()
    ss.workout = replace_exercise(DF, ss.profile, ss.workout, idx, new_id, model=model, use_ml=model is not None)
    ss.flash = idx
    st.toast(f"Swapped to {DF.set_index('exercise_id').loc[new_id, 'exercise_name']}", icon="🔁")


def why_bars(item) -> str:
    labels = {"muscle": "Target muscle", "equipment": "Equipment", "level": "Experience fit", "goal": "Goal fit", "pref": "Your preference"}
    rows = "".join(f'<div class="why"><span>{labels[k]}</span><div class="t"><i style="width:{v * 100:.0f}%"></i></div><span>{v:.2f}</span></div>' for k, v in item.parts.items())
    return h(rows)


def page_workout():
    w, p = ss.workout, ss.profile
    if w is None or p is None:
        st.subheader("No workout yet")
        st.markdown('<p class="sub">Answer a few questions and your plan will appear here.</p>', unsafe_allow_html=True)
        st.button("Create my workout", key="empty_cta", type="primary", on_click=goto, args=("create",))
        return
    title = f"{esc(p.name)}, your" if p.name else "Your"
    st.markdown(f'<h1 style="font-size:clamp(30px,5vw,48px);margin-bottom:6px">{title} {p.muscle_group} workout</h1>', unsafe_allow_html=True)
    chips = [C.GOALS[p.goal][0], pretty(p.level), pretty(p.location), f"{p.duration} min", f"{pretty(p.intensity)} intensity"]
    if p.emphasis:
        chips.append("Emphasis: " + ", ".join(pretty(e) for e in p.emphasis))
    st.markdown('<div class="sum">' + "".join(f"<span>{esc(c)}</span>" for c in chips) + "</div>", unsafe_allow_html=True)
    for m in w.messages:
        st.markdown(f'<div class="warn">{esc(m)}</div>', unsafe_allow_html=True)
    with st.expander("How we got here (filter funnel)"):
        top = max(w.funnel[0][1], 1)
        st.markdown(h("".join(f'<div class="fun"><span>{esc(l)}</span><div class="t"><i style="width:{max(n / top * 100, 1.5):.0f}%"></i></div><span>{n}</span></div>' for l, n in w.funnel)), unsafe_allow_html=True)
        st.caption("Safety exclusions run first and can never be overridden by a score or by machine learning.")
    if not w.items:
        st.markdown('<div class="note"><b>No suitable match found.</b> Nothing in the catalogue fits this muscle group with your level, equipment and exclusions. '
                    'Try adding equipment, removing an exclusion or choosing another muscle group.</div>', unsafe_allow_html=True)
        st.button("Change answers", key="nomatch_edit", type="primary", on_click=goto, args=("create",))
        return
    components.html(stats_component([(len(w.items), "exercises"), (w.est_minutes, "estimated minutes"), (w.total_sets, "working sets")]), height=120)
    if w.used_ml:
        st.markdown('<div class="note">A validated feedback model is contributing 20% of the ranking score. Safety filters still come first.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Warm-up:</b> about 5 minutes of easy movement, then 1-2 light sets of your first exercise.</div>', unsafe_allow_html=True)

    for i, it in enumerate(w.items):
        r = it.row
        with st.container(border=True, key=f"ex_{i}"):
            if ss.flash == i:
                st.markdown('<div class="st-key-flashcard" style="height:0"></div>', unsafe_allow_html=True)
            a, b = st.columns([4, 1.6], vertical_alignment="center")
            a.markdown(f'<div class="exn"><span class="badge">{i + 1}</span><div><h3>{esc(r["exercise_name"])}</h3>'
                       f'<div class="meta">{pretty(r["muscle_emphasis"])} · {C.EQUIPMENT_LABELS[r["equipment"]]} · {pretty(r["exercise_type"])}</div></div></div>', unsafe_allow_html=True)
            b.markdown(f'<div class="rx">{it.sets} × {esc(it.reps)}<small>rest {it.rest}s</small></div>', unsafe_allow_html=True)
            with st.expander("Technique, why this exercise, swap"):
                tags = f'<span class="tag pr">Primary: {pretty(r["primary_muscles"])}</span>' + "".join(f'<span class="tag">{pretty(s)}</span>' for s in r["secondary_list"])
                if not r["reviewed"]:
                    tags += '<span class="tag">Unreviewed data</span>'
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown(f"**How to.** {r['instructions']}")
                st.markdown(f"**Common mistakes.** {r['common_mistakes']}")
                st.markdown(f"**Caution.** {r['contraindication_notes']}")
                st.markdown("**Why it was chosen**")
                st.markdown(why_bars(it), unsafe_allow_html=True)
                if it.ml_prob is not None:
                    st.caption(f"Feedback model: {it.ml_prob:.0%} chance you rate this 4+ stars.")
                alts = get_alternatives(DF, p, SIM, w, i, k=4, model=model_in_use(), use_ml=ss.use_ml)
                st.markdown("**Swap with**")
                if not alts:
                    st.caption("No suitable alternative found within your constraints.")
                for al in alts:
                    c1, c2 = st.columns([4, 1], vertical_alignment="center")
                    badge = "Reviewed pairing" if al["source"] == "curated" else "ML similar"
                    c1.markdown(f'<span class="tag {"pr" if al["source"] == "curated" else "ml"}">{badge}</span> **{esc(al["name"])}** '
                                f'<span class="meta">{C.EQUIPMENT_LABELS[al["equipment"]]} · {al["similarity"]:.0%} similar</span>', unsafe_allow_html=True)
                    c2.button("Swap", key=f"swap_{i}_{al['id']}", on_click=swap, args=(i, al["id"]))

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Generate another plan", key="regen", type="primary", use_container_width=True):
        generate(jitter=0.15, seed=int(time.time()) % 100000, animate=False)
    c2.download_button("Download .txt", workout_to_text(w, p), "workout-plan.txt", use_container_width=True)
    c3.download_button("Download .csv", workout_to_csv(w), "workout-plan.csv", use_container_width=True)
    c4.button("Change answers", key="edit", on_click=goto, args=("create",), use_container_width=True)
    ss.flash = None

    st.write("")
    with st.container(border=True):
        st.subheader("How did it go?")
        st.markdown('<p class="sub">Optional. Ratings help the app learn which exercises people find useful. '
                    'We save only your ratings and the exercise attributes, never your name, age or health answers.</p>', unsafe_allow_html=True)
        if ss.fb_sent:
            st.success("Thanks. Your feedback was saved to this app's local feedback file.")
        else:
            with st.form("feedback"):
                f1, f2 = st.columns(2)
                comp = f1.radio("Session", ["completed", "partial", "not_completed"], format_func=pretty, horizontal=True)
                diff = f2.radio("Difficulty", ["too_easy", "appropriate", "too_hard"], format_func=pretty, horizontal=True)
                ratings = {}
                for i, it in enumerate(w.items):
                    cc = st.columns([3, 2])
                    cc[0].write(f"{i + 1}. {it.row['exercise_name']}")
                    ratings[i] = cc[1].feedback("stars", key=f"star_{i}")
                if st.form_submit_button("Save feedback", type="primary"):
                    rows = []
                    for i, it in enumerate(w.items):
                        if ratings[i] is None:
                            continue
                        r = it.row
                        rows.append({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session_id": ss.session_id,
                                     "exercise_id": r["exercise_id"], "duration": p.duration, "user_level": p.level, "goal": p.goal,
                                     "intensity": p.intensity, "muscle_group": r["muscle_group"], "movement_pattern": r["movement_pattern"],
                                     "equipment": r["equipment"], "exercise_type": r["exercise_type"], "difficulty": r["difficulty"],
                                     "level_gap": C.LEVELS.index(p.level) - r["min_level_idx"], "rule_score": r["rule_score"],
                                     "completion": comp, "perceived_difficulty": diff, "rating": ratings[i] + 1, "is_simulated": False})
                    if rows:
                        append_feedback(rows)
                        ss.fb_sent = True
                        st.rerun()
                    else:
                        st.warning("Rate at least one exercise to save feedback.")


def page_ml():
    st.title("ML Lab")
    st.markdown('<p class="sub">Machine learning here has two jobs. One works from day one. The other waits until there is enough real feedback to prove it helps.</p>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Exercise similarity", "Feedback model"])
    with t1:
        st.markdown("**Content-based model (TF-IDF + cosine similarity).** Every exercise becomes a vector of muscles, pattern, equipment, type and difficulty. "
                    "Similar vectors mean similar exercises. It powers swaps and keeps sessions from repeating the same movement.")
        xy = SIM.coords()
        pts = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "muscle group": DF.muscle_group.map(pretty)})
        st.scatter_chart(pts, x="x", y="y", color="muscle group", height=320)
        st.caption("Each dot is an exercise, projected to 2D with TruncatedSVD. Neighbours are alike.")
        pick = st.selectbox("Find exercises similar to", DF.exercise_id, format_func=lambda e: DF.set_index("exercise_id").loc[e, "exercise_name"])
        by_id = DF.set_index("exercise_id")
        res = pd.DataFrame([{"Exercise": by_id.loc[e, "exercise_name"], "Group": pretty(by_id.loc[e, "muscle_group"]),
                             "Equipment": C.EQUIPMENT_LABELS[by_id.loc[e, "equipment"]], "Similarity": s} for e, s in SIM.similar(pick, 8)])
        st.dataframe(res, hide_index=True, use_container_width=True,
                     column_config={"Similarity": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    with t2:
        st.markdown("**Feedback model (Logistic Regression and Gradient Boosting).** Predicts whether someone will rate an exercise 4+ stars. "
                    "It is cross-validated against the rule-based baseline using grouped folds, and it only affects plans if it wins. "
                    "It can only re-order exercises that already passed the safety filters.")
        real = load_feedback()
        have = int(real["rating"].notna().sum()) if "rating" in real else 0
        st.progress(min(have / C.MIN_FEEDBACK_ROWS, 1.0), text=f"Real ratings collected: {have} of {C.MIN_FEEDBACK_ROWS} needed")
        src = st.radio("Training data", ["Real feedback", "Simulated demo data (pipeline test only)"], horizontal=True)
        simulated = src.startswith("Simulated")
        if simulated:
            st.markdown('<div class="warn">Simulated ratings come from a made-up formula plus noise. A model trained on them says nothing about real exercise quality and is never activated.</div>', unsafe_allow_html=True)
        path = C.SIMULATED_FEEDBACK_FILE if simulated else C.FEEDBACK_FILE
        if st.button("Train and evaluate", type="primary", key="train"):
            data = load_feedback(path)
            if data.empty:
                st.warning("No data found for this source yet.")
            else:
                with st.spinner("Cross-validating against the rule-based baseline…"):
                    train_feedback_model(data, simulated=simulated)
                st.cache_resource.clear()
        rep = load_report(simulated)
        if rep:
            st.markdown(f"**Last run:** {rep['trained_at']} · {rep['n_rows']} rows · {rep.get('n_sessions', 0)} sessions")
            if "models" in rep:
                m1, m2, m3 = st.columns(3)
                m1.metric("Rule baseline AUC", f"{rep['baseline_auc']:.3f}")
                m2.metric(f"Best model ({pretty(rep['best'])}) AUC", f"{rep['models'][rep['best']]['auc_mean']:.3f}", f"{rep['auc_gain']:+.3f}")
                m3.metric("Folds", rep["cv_folds"])
                st.dataframe(pd.DataFrame(rep["models"]).T.rename(columns={"auc_mean": "AUC mean", "auc_std": "AUC std"}), use_container_width=True)
                st.bar_chart(pd.DataFrame(rep["top_features"]).set_index("feature"), height=240)
            (st.success if rep["eligible"] and not simulated else st.info)(rep["reason"])
        if not simulated:
            model_ok = active_model() is not None
            ss.use_ml = st.toggle("Use the validated model when ranking exercises", value=ss.use_ml and model_ok, disabled=not model_ok)
            st.caption("Active: the model contributes 20% of the score." if model_ok else "Inactive: rules rank exercises until a model passes validation on real data.")
            if have and st.button("Delete my saved feedback", key="del_fb"):
                C.FEEDBACK_FILE.unlink(missing_ok=True)
                st.rerun()


def page_about():
    st.title("About & safety")
    st.markdown("""
**What it does.** Builds a structured single-muscle workout from a curated catalogue, with sets, reps, rest, technique cues and swaps.

**How recommendations are made.**
1. Safety and exclusion filters (avoided exercises, protected body areas, conservative difficulty when you report pain).
2. Rule filters for muscle group, equipment, location and experience level.
3. Weighted ranking: target muscle 35%, equipment 20%, experience 20%, goal 15%, preference 10%.
4. Composition that avoids near-duplicate movements (cosine similarity) and fits your time.
5. Optional: a feedback model, used only after it beats the rules on real data.

**Data limits.** The catalogue is compiled from general resistance-training knowledge and every entry is marked unreviewed. Muscle emphasis labels are
training emphasis, not isolation. Set and rep ranges are starting points, not ideal prescriptions.

**Privacy.** The plan stays in your session. Optional ratings save only exercise attributes and your rating to a local CSV, with an anonymous session id.
Name, age, pain, injury and medical answers are never saved or used for training. You can delete saved feedback in the ML Lab.
A shared CSV is not suitable for many concurrent users; use a database for a public deployment.

**Safety.** This app is educational fitness guidance, not medical advice. It does not diagnose conditions, prevent injuries or replace a qualified coach or
healthcare professional. If you have pain, an injury or a medical restriction, talk to a doctor or physiotherapist first, and stop if anything hurts.
""")


header()
{"home": page_home, "create": page_create, "workout": page_workout, "ml": page_ml, "about": page_about}[ss.page]()
