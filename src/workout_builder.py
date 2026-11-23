"""Workout composition, duration fitting, alternatives and export (spec 6.3 and 6.4)."""
from __future__ import annotations
import io
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import config as C
from .ml_ranker import ContentSimilarity
from .recommender import get_candidates
from .safety_filters import safety_messages
from .validators import UserProfile

TYPE_ORDER = {"compound": 0, "isolation": 1, "stability": 2, "isometric": 3}


@dataclass
class PlanItem:
    row: dict
    sets: int
    reps: str
    rest: int
    minutes: float
    score: float
    parts: dict
    ml_prob: float | None = None


@dataclass
class Workout:
    items: list[PlanItem]
    est_minutes: int = 0
    messages: list[str] = field(default_factory=list)
    over_time: bool = False
    funnel: list = field(default_factory=list)
    used_ml: bool = False

    @property
    def total_sets(self) -> int:
        return sum(i.sets for i in self.items)


def prescribe(row: dict, p: UserProfile, score: float, ml_prob=None) -> PlanItem:
    s0, s1 = int(row["sets_min"]), int(row["sets_max"])
    sets = s0 if p.intensity == "low" else s1 if p.intensity == "high" else (s0 + s1 + 1) // 2
    lo, hi = int(row["reps_min"]), int(row["reps_max"])
    span = hi - lo
    a, b = {"strength": (lo, lo + span / 3), "fat_loss": (hi - span / 3, hi),
            "muscle_gain": (lo + span / 4, hi - span / 4), "general_fitness": (lo + span / 4, hi)}[p.goal]
    a, b = int(round(a)), int(round(b))
    reps = f"{a}" if a == b else f"{a}-{b}"
    unit = " sec" if row["rep_unit"] == "seconds" else ""
    rest = int(np.clip(row["rest_seconds"] + (30 if p.goal == "strength" else -15 if p.goal == "fat_loss" else 0), 30, 180))
    work = (a + b) / 2 if row["rep_unit"] == "seconds" else (a + b) / 2 * 3.5
    minutes = sets * (work + rest) / 60 + C.TRANSITION_MINUTES
    parts = {k: float(row[f"s_{k}"]) for k in ("muscle", "equipment", "level", "goal", "pref")}
    return PlanItem(row, sets, reps + unit, rest, minutes, float(score), parts, None if pd.isna(ml_prob) else float(ml_prob))


def _finish(items: list[PlanItem], p: UserProfile, funnel, used_ml: bool) -> Workout:
    est = int(round(C.WARMUP_MINUTES + sum(i.minutes for i in items)))
    msgs = safety_messages(p)
    over = est > p.duration + 1
    if over:
        msgs.append(f"The shortest sensible plan takes about {est} min, longer than your {p.duration} min. "
                    "Choose more time or accept a shorter, lighter session.")
    return Workout(items, est, msgs, over, funnel, used_ml)


def build_workout(df: pd.DataFrame, p: UserProfile, sim: ContentSimilarity, model=None, use_ml=False,
                  jitter: float = 0.0, seed: int | None = None) -> Workout:
    c, funnel = get_candidates(df, p, model, use_ml)
    active_ml = bool(use_ml and model is not None)
    if c.empty:
        return Workout([], 0, safety_messages(p), False, funnel, active_ml)
    rng = np.random.default_rng(seed)
    c["pick"] = c["final_score"] + rng.random(len(c)) * jitter
    budget = p.duration - C.WARMUP_MINUTES
    chosen: list[PlanItem] = []
    used = 0.0
    pool = c.copy()
    while len(chosen) < C.MAX_EXERCISES and not pool.empty:
        best_i, best_val = None, -9
        chosen_ids = [i.row["exercise_id"] for i in chosen]
        chosen_pat = {i.row["movement_pattern"] for i in chosen}
        for idx, r in pool.iterrows():
            div = max((sim.sim(r.exercise_id, e) for e in chosen_ids), default=0.0)   # MMR: avoid near-duplicates
            pen = 0.15 if r.movement_pattern in chosen_pat else 0.0
            val = C.MMR_LAMBDA * r.pick - (1 - C.MMR_LAMBDA) * div - pen
            if val > best_val:
                best_i, best_val = idx, val
        r = pool.loc[best_i]
        item = prescribe(r.to_dict(), p, r.final_score, r.ml_prob)
        pool = pool.drop(best_i)
        if used + item.minutes > budget and len(chosen) >= 2:
            continue                                     # does not fit; try a shorter one
        chosen.append(item)
        used += item.minutes
    chosen.sort(key=lambda i: (TYPE_ORDER[i.row["exercise_type"]], -i.score))
    return _finish(chosen, p, funnel, active_ml)


def get_alternatives(df, p, sim, workout: Workout, idx: int, k: int = 4, model=None, use_ml=False) -> list[dict]:
    """Reviewed alternative_ids first, then content-similar exercises. Always inside the user's constraints."""
    c, _ = get_candidates(df, p, model, use_ml)
    used = {i.row["exercise_id"] for i in workout.items}
    c = c[~c.exercise_id.isin(used)]
    if c.empty:
        return []
    cur = workout.items[idx].row
    allowed = set(c.exercise_id)
    curated = [a for a in cur["alt_list"] if a in allowed]
    out = [{"id": a, "similarity": sim.sim(cur["exercise_id"], a), "source": "curated"} for a in curated]
    for eid, s in sim.similar(cur["exercise_id"], k=k, allowed=allowed - set(curated)):
        out.append({"id": eid, "similarity": s, "source": "similar"})
    by_id = c.set_index("exercise_id")
    for o in out:
        o["name"] = by_id.loc[o["id"], "exercise_name"]
        o["equipment"] = by_id.loc[o["id"], "equipment"]
    return out[:k]


def replace_exercise(df, p, workout: Workout, idx: int, new_id: str, model=None, use_ml=False) -> Workout:
    c, funnel = get_candidates(df, p, model, use_ml)
    r = c[c.exercise_id == new_id]
    if r.empty:
        return workout                                   # never insert something that fails the constraints
    r = r.iloc[0]
    items = list(workout.items)
    items[idx] = prescribe(r.to_dict(), p, r.final_score, r.ml_prob)
    return _finish(items, p, workout.funnel, workout.used_ml)


def workout_to_text(w: Workout, p: UserProfile) -> str:
    lines = ["AI WORKOUT PLAN", f"Goal: {C.GOALS[p.goal][0]} | Level: {p.level} | Target: {p.muscle_group} | "
             f"{p.duration} min requested | ~{w.est_minutes} min planned", "",
             "Warm-up: about 5 minutes of easy movement, then 1-2 light sets of the first exercise.", ""]
    for n, i in enumerate(w.items, 1):
        r = i.row
        lines += [f"{n}. {r['exercise_name']} ({C.EQUIPMENT_LABELS[r['equipment']]})",
                  f"   {i.sets} x {i.reps}, rest {i.rest}s | primary: {r['primary_muscles']}",
                  f"   How: {r['instructions']}", f"   Avoid: {r['common_mistakes']}", f"   Note: {r['contraindication_notes']}", ""]
    lines.append("Educational fitness guidance, not medical advice. Exercise data is unreviewed until checked by a qualified professional.")
    return "\n".join(lines)


def workout_to_csv(w: Workout) -> str:
    rows = [{"order": n, "exercise_id": i.row["exercise_id"], "exercise": i.row["exercise_name"],
             "equipment": i.row["equipment"], "sets": i.sets, "reps": i.reps, "rest_seconds": i.rest,
             "primary_muscles": i.row["primary_muscles"]} for n, i in enumerate(w.items, 1)]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()
