"""Load and validate the curated exercise catalogue."""
from __future__ import annotations
import pandas as pd
from . import config as C

REQUIRED = [
    "exercise_id", "exercise_name", "muscle_group", "primary_muscles", "secondary_muscles",
    "muscle_emphasis", "movement_pattern", "equipment", "location", "level", "difficulty",
    "exercise_type", "instructions", "common_mistakes", "sets_min", "sets_max", "reps_min",
    "reps_max", "rep_unit", "rest_seconds", "duration_minutes", "contraindication_notes",
    "alternative_ids", "source", "reviewed",
]
NO_EMPTY = ["exercise_id", "exercise_name", "muscle_group", "primary_muscles", "movement_pattern",
            "equipment", "location", "level", "difficulty", "exercise_type", "instructions"]


class DataValidationError(ValueError):
    """Raised with a readable list of every problem found in the catalogue."""


def split_values(value, lower: bool = True) -> list[str]:
    """Split 'a;b' into ['a','b']. Never uses eval(); CSV content is data only."""
    if value is None or pd.isna(value):
        return []
    items = [i.strip() for i in str(value).split(";") if i.strip()]
    return [i.lower() for i in items] if lower else items


def validate_exercises(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if df["exercise_id"].duplicated().any():
        problems.append(f"Duplicate exercise_id: {df.loc[df.exercise_id.duplicated(), 'exercise_id'].tolist()}")
    if df["exercise_name"].str.lower().duplicated().any():
        problems.append("Duplicate exercise names (check spelling variants).")
    for col in NO_EMPTY:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            problems.append(f"Empty values in required column '{col}'.")
    allowed = {"muscle_group": C.MUSCLE_GROUPS, "equipment": C.EQUIPMENT, "difficulty": C.DIFFICULTIES,
               "exercise_type": C.EXERCISE_TYPES, "rep_unit": C.REP_UNITS}
    for col, ok in allowed.items():
        bad = sorted(set(df[col].dropna().astype(str)) - set(ok))
        if bad:
            problems.append(f"Unexpected values in '{col}': {bad}")
    for col, ok in {"location": C.LOCATIONS, "level": C.LEVELS}.items():
        bad = sorted({t for v in df[col] for t in split_values(v)} - set(ok))
        if bad:
            problems.append(f"Unexpected values in '{col}': {bad}")
    for lo, hi in (("sets_min", "sets_max"), ("reps_min", "reps_max")):
        if (df[lo] > df[hi]).any():
            problems.append(f"{lo} is greater than {hi} in some rows.")
    ids = set(df["exercise_id"])
    bad_alt = sorted({a for v in df["alternative_ids"] for a in split_values(v, lower=False)} - ids)
    if bad_alt:
        problems.append(f"alternative_ids reference unknown exercises: {bad_alt}")
    return problems


def load_exercises(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.EXERCISE_FILE)
    problems = validate_exercises(df)
    if problems:
        raise DataValidationError("Exercise catalogue problems:\n- " + "\n- ".join(problems))
    df = df.drop_duplicates(subset=["exercise_id"]).dropna(subset=["exercise_id", "exercise_name"]).reset_index(drop=True)
    df["secondary_list"] = df["secondary_muscles"].apply(split_values)
    df["location_list"] = df["location"].apply(split_values)
    df["level_list"] = df["level"].apply(split_values)
    df["alt_list"] = df["alternative_ids"].apply(lambda v: split_values(v, lower=False))
    df["min_level_idx"] = df["level_list"].apply(lambda l: min(C.LEVELS.index(x) for x in l))
    df["difficulty_idx"] = df["difficulty"].map({d: i for i, d in enumerate(C.DIFFICULTIES)})
    df["reviewed"] = df["reviewed"].astype(str).str.lower().eq("true")
    return df
