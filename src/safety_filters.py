"""Safety and exclusion filters. These run BEFORE scoring and cannot be overridden by a score or by ML."""
from __future__ import annotations
import pandas as pd
from . import config as C
from .validators import UserProfile, tokens


def apply_safety_exclusions(df: pd.DataFrame, p: UserProfile) -> pd.DataFrame:
    out = df
    # 1. exercises the user explicitly wants to avoid (matches name or movement pattern)
    for t in tokens(p.avoid_text):
        t_clean = t.replace("-", " ")
        hay = out.exercise_name.str.lower().str.replace("-", " ") + " " + out.movement_pattern.str.replace("_", " ")
        out = out[~hay.str.contains(t_clean, regex=False)]
    # 2. body areas to protect -> drop exercises whose caution note names that area
    for area in p.protect_areas:
        for kw in C.PROTECT_AREAS.get(area, []):
            out = out[~out.contraindication_notes.str.lower().str.contains(kw, regex=False)]
    # 3. current pain / injury / medical restriction -> keep only conservative difficulty
    if p.pain or p.injury or p.medical:
        out = out[out.difficulty != "high"]
    return out


def safety_messages(p: UserProfile) -> list[str]:
    msgs = []
    if p.pain or p.injury or p.medical:
        msgs.append("You reported pain, an injury or a medical restriction. This app cannot diagnose or treat it. "
                    "Please get advice from a doctor or physiotherapist before following any plan, and stop if anything hurts.")
    if p.pain and not p.protect_areas:
        msgs.append("Tip: choose the affected area under 'Areas to protect' so related exercises are removed.")
    return msgs
