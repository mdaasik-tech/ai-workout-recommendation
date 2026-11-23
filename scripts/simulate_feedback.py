"""Generate SIMULATED feedback to test the ML pipeline end to end.

WARNING: these ratings come from a made-up rule plus noise, not from people. A model trained on them learns the
made-up rule and says nothing about real exercise quality. The app never activates a model trained on this file.
Usage: python scripts/simulate_feedback.py --sessions 160
"""
import argparse, sys, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.data_loader import load_exercises
from src.ml_ranker import ContentSimilarity, FEEDBACK_COLUMNS
from src.validators import UserProfile
from src.workout_builder import build_workout
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--sessions", type=int, default=160); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
df = load_exercises(); sim = ContentSimilarity(df)
rows = []
for s in range(a.sessions):
    loc = rng.choice(C.LOCATIONS)
    pool = C.EQUIPMENT if loc == "gym" else ["dumbbells", "resistance_bands", "bodyweight", "pull_up_bar"]
    eq = list(rng.choice(pool, size=rng.integers(2, len(pool) + 1), replace=False))
    p = UserProfile(goal=rng.choice(list(C.GOALS)), level=rng.choice(C.LEVELS), muscle_group=rng.choice(C.MUSCLE_GROUPS),
                    location=loc, equipment=eq, duration=int(rng.choice(C.DURATIONS)), intensity=rng.choice(C.INTENSITIES))
    w = build_workout(df, p, sim, jitter=0.3, seed=int(rng.integers(1e9)))
    sid = uuid.uuid4().hex[:12]
    for it in w.items:
        r = it.row
        z = 3.3 + 4.0 * (it.score - 0.8) - 0.9 * r["difficulty_idx"] * (p.level == "beginner")
        z += 0.5 * (p.goal == "strength" and r["exercise_type"] == "compound") - 0.3 * (p.intensity == "high" and r["exercise_type"] == "isolation")
        z -= 0.5 * (p.level == "advanced" and r["difficulty"] == "low") + rng.normal(0, 0.8)
        rating = int(np.clip(round(z), 1, 5))
        rows.append({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session_id": sid, "exercise_id": r["exercise_id"],
                     "duration": p.duration, "user_level": p.level, "goal": p.goal, "intensity": p.intensity,
                     "muscle_group": r["muscle_group"], "movement_pattern": r["movement_pattern"], "equipment": r["equipment"],
                     "exercise_type": r["exercise_type"], "difficulty": r["difficulty"],
                     "level_gap": C.LEVELS.index(p.level) - r["min_level_idx"], "rule_score": it.score,
                     "completion": "completed" if rating >= 3 else "partial",
                     "perceived_difficulty": "too_hard" if z < 2.3 else "appropriate", "rating": rating, "is_simulated": True})
out = pd.DataFrame(rows).reindex(columns=FEEDBACK_COLUMNS)
C.SIMULATED_FEEDBACK_FILE.parent.mkdir(exist_ok=True)
out.to_csv(C.SIMULATED_FEEDBACK_FILE, index=False)
print(f"Wrote {len(out)} SIMULATED rows from {a.sessions} sessions to {C.SIMULATED_FEEDBACK_FILE}")
