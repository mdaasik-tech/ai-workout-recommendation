"""Rule-based filtering + weighted ranking (spec sections 6.1 and 6.2)."""
from __future__ import annotations
import pandas as pd
from . import config as C
from .ml_ranker import feature_frame, predict_like_probability
from .safety_filters import apply_safety_exclusions
from .validators import UserProfile, tokens

TARGET_DIFFICULTY = {"beginner": 0.0, "intermediate": 1.0, "advanced": 1.5}
GOAL_TYPE_FIT = {
    "strength":        {"compound": 1.0, "isolation": 0.4, "isometric": 0.3, "stability": 0.3},
    "muscle_gain":     {"compound": 0.85, "isolation": 0.9, "isometric": 0.4, "stability": 0.4},
    "general_fitness": {"compound": 0.9, "isolation": 0.7, "isometric": 0.7, "stability": 0.8},
    "fat_loss":        {"compound": 1.0, "isolation": 0.5, "isometric": 0.6, "stability": 0.6},
}


def filter_candidates(df: pd.DataFrame, p: UserProfile) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Safety filters first, then muscle -> equipment -> location -> level. Returns the funnel for explainability."""
    funnel = [("Exercise catalogue", len(df))]
    c = apply_safety_exclusions(df, p)
    funnel.append(("After safety & exclusions", len(c)))
    c = c[c.muscle_group == p.muscle_group]
    funnel.append((f"Target: {p.muscle_group}", len(c)))
    c = c[c.equipment.isin(p.equipment)]
    funnel.append(("Matches your equipment", len(c)))
    c = c[c.location_list.apply(lambda l: p.location in l)]
    funnel.append((f"Suitable for {p.location}", len(c)))
    c = c[c.level_list.apply(lambda l: p.level in l)]
    funnel.append((f"Suitable for {p.level}", len(c)))
    return c.copy(), funnel


def score_candidates(c: pd.DataFrame, p: UserProfile) -> pd.DataFrame:
    """Every component is normalised to 0..1 (spec 6.2)."""
    if c.empty:
        return c.assign(s_muscle=[], s_equipment=[], s_level=[], s_goal=[], s_pref=[], rule_score=[])
    emph, prefs = set(p.emphasis), tokens(p.prefer_text)
    other_gear = len(set(p.equipment) - {"bodyweight"}) > 0
    c["s_muscle"] = c.muscle_emphasis.apply(lambda e: 1.0 if (not emph or e in emph) else 0.6)
    c["s_equipment"] = c.equipment.apply(lambda e: 0.85 if (e == "bodyweight" and other_gear) else 1.0)
    target = TARGET_DIFFICULTY[p.level]
    in_list = c.level_list.apply(lambda l: 1.0 if (len(l) == 1 or l[0] == p.level) else 0.85)
    c["s_level"] = 0.5 * in_list + 0.5 * (1 - (c.difficulty_idx - target).abs() / 2).clip(0, 1)
    fit = GOAL_TYPE_FIT[p.goal]
    c["s_goal"] = c.exercise_type.map(fit) + (0.1 if p.goal == "strength" else 0) * (c.reps_min <= 8)
    c["s_goal"] = c["s_goal"].clip(0, 1)
    c["s_pref"] = c.exercise_name.str.lower().apply(lambda n: 1.0 if any(t in n for t in prefs) else 0.5)
    w = C.WEIGHTS
    c["rule_score"] = (w["muscle"] * c.s_muscle + w["equipment"] * c.s_equipment + w["level"] * c.s_level
                       + w["goal"] * c.s_goal + w["preference"] * c.s_pref)
    return c


def get_candidates(df: pd.DataFrame, p: UserProfile, model=None, use_ml: bool = False):
    c, funnel = filter_candidates(df, p)
    c = score_candidates(c, p)
    c["ml_prob"] = float("nan")
    c["final_score"] = c["rule_score"]
    if use_ml and model is not None and not c.empty:
        c["ml_prob"] = predict_like_probability(model, feature_frame(c, p))
        c["final_score"] = (1 - C.ML_BLEND) * c["rule_score"] + C.ML_BLEND * c["ml_prob"]
    return c.sort_values("final_score", ascending=False), funnel
