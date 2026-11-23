"""Machine Learning components (scikit-learn).

1. ContentSimilarity  - TF-IDF + cosine similarity over exercise attributes. Works from day one, needs no
                        user data. Used to rank alternatives, to avoid near-duplicate exercises in a session
                        and to draw the exercise map.
2. Feedback model     - Logistic Regression / Gradient Boosting trained on CONSENTED rating data. It is only
                        allowed to influence ranking after it beats the rule-based baseline in cross-validation
                        on real data. It re-orders safe candidates; it can never add an excluded exercise.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from . import config as C

CAT = ["user_level", "goal", "intensity", "muscle_group", "movement_pattern", "equipment", "exercise_type", "difficulty"]
NUM = ["level_gap", "rule_score"]
FEEDBACK_COLUMNS = ["timestamp", "session_id", "exercise_id", "duration", *CAT, *NUM,
                    "completion", "perceived_difficulty", "rating", "is_simulated"]


# ---------------------------------------------------------------- content similarity
class ContentSimilarity:
    def __init__(self, df: pd.DataFrame):
        self.ids = df["exercise_id"].tolist()
        self.pos = {e: i for i, e in enumerate(self.ids)}
        docs = df.apply(self._doc, axis=1)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
        self.X = self.vectorizer.fit_transform(docs)
        self.matrix = cosine_similarity(self.X)

    @staticmethod
    def _doc(r) -> str:
        t = [f"grp_{r.muscle_group}"] * 2 + [f"pm_{r.primary_muscles}"] + [f"em_{r.muscle_emphasis}"] * 2
        t += [f"sec_{s}" for s in r.secondary_list] + [f"mp_{r.movement_pattern}"] * 3
        t += [f"eq_{r.equipment}", f"ty_{r.exercise_type}", f"df_{r.difficulty}"]
        return " ".join(t)

    def sim(self, a: str, b: str) -> float:
        return float(self.matrix[self.pos[a], self.pos[b]])

    def similar(self, exercise_id: str, k: int = 5, allowed: set | None = None) -> list[tuple[str, float]]:
        row = self.matrix[self.pos[exercise_id]]
        order = np.argsort(-row)
        out = []
        for i in order:
            eid = self.ids[i]
            if eid == exercise_id or (allowed is not None and eid not in allowed):
                continue
            out.append((eid, float(row[i])))
            if len(out) == k:
                break
        return out

    def coords(self) -> np.ndarray:
        return TruncatedSVD(n_components=2, random_state=0).fit_transform(self.X)


# ---------------------------------------------------------------- feedback data
def feature_frame(c: pd.DataFrame, p) -> pd.DataFrame:
    """Features shared by training and inference. Deliberately excludes pain/injury/medical details."""
    f = pd.DataFrame({
        "user_level": p.level, "goal": p.goal, "intensity": p.intensity,
        "muscle_group": c["muscle_group"].values, "movement_pattern": c["movement_pattern"].values,
        "equipment": c["equipment"].values, "exercise_type": c["exercise_type"].values,
        "difficulty": c["difficulty"].values,
        "level_gap": C.LEVELS.index(p.level) - c["min_level_idx"].values,
        "rule_score": c["rule_score"].values,
    })
    return f[CAT + NUM]


def append_feedback(rows: list[dict], path=C.FEEDBACK_FILE) -> int:
    """Single-user prototype storage. For a public multi-user app move this to a database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=FEEDBACK_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(df)


def load_feedback(path=C.FEEDBACK_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(path)


# ---------------------------------------------------------------- feedback model
def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", NUM)])
    return Pipeline([("pre", pre), ("model", estimator)])


def _candidates() -> dict:
    return {
        "logistic_regression": lambda: _pipeline(LogisticRegression(max_iter=2000, C=0.5)),
        "gradient_boosting": lambda: _pipeline(GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=0)),
    }


def _top_features(pipe: Pipeline, n: int = 8) -> list[dict]:
    names = pipe.named_steps["pre"].get_feature_names_out()
    m = pipe.named_steps["model"]
    vals = m.coef_[0] if hasattr(m, "coef_") else m.feature_importances_
    order = np.argsort(-np.abs(vals))[:n]
    return [{"feature": str(names[i]).replace("cat__", "").replace("num__", ""), "weight": float(vals[i])} for i in order]


def train_feedback_model(df: pd.DataFrame, simulated: bool = False) -> dict:
    """Cross-validate against the rule-based baseline and decide whether the model may be used."""
    report = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "simulated" if simulated else "real", "n_rows": int(len(df)),
              "eligible": False, "active": False, "reason": ""}
    df = df.dropna(subset=CAT + NUM + ["rating"]).copy()
    y = (df["rating"] >= 4).astype(int)
    X = df[CAT + NUM]
    report.update(n_rows=int(len(df)), n_sessions=int(df["session_id"].nunique()),
                  positive_rate=float(y.mean()) if len(y) else 0.0)
    if len(df) < C.MIN_FEEDBACK_ROWS:
        report["reason"] = f"Need at least {C.MIN_FEEDBACK_ROWS} rated exercises (have {len(df)})."
        return _save_report(report, simulated)
    if min(y.sum(), len(y) - y.sum()) < C.MIN_CLASS_ROWS:
        report["reason"] = f"Need at least {C.MIN_CLASS_ROWS} liked and {C.MIN_CLASS_ROWS} not-liked ratings."
        return _save_report(report, simulated)
    groups = df["session_id"].astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {k: [] for k in _candidates()}
    base = []
    for tr, te in cv.split(X, y, groups):
        if y.iloc[te].nunique() < 2 or y.iloc[tr].nunique() < 2:
            continue
        base.append(roc_auc_score(y.iloc[te], X["rule_score"].iloc[te]))
        for name, make in _candidates().items():
            pipe = make().fit(X.iloc[tr], y.iloc[tr])
            scores[name].append(roc_auc_score(y.iloc[te], pipe.predict_proba(X.iloc[te])[:, 1]))
    if not base:
        report["reason"] = "Not enough variety across sessions to cross-validate."
        return _save_report(report, simulated)
    summary = {k: {"auc_mean": float(np.mean(v)), "auc_std": float(np.std(v))} for k, v in scores.items()}
    best = max(summary, key=lambda k: summary[k]["auc_mean"])
    base_auc = float(np.mean(base))
    final = _candidates()[best]().fit(X, y)
    acc = accuracy_score(y, final.predict(X))
    report.update(models=summary, best=best, baseline_auc=base_auc, train_accuracy=float(acc),
                  top_features=_top_features(final), cv_folds=len(base))
    gain = summary[best]["auc_mean"] - base_auc
    report["auc_gain"] = float(gain)
    auc, sd = summary[best]["auc_mean"], summary[best]["auc_std"]
    if gain >= C.MIN_AUC_GAIN and gain > sd and auc >= C.MIN_ABS_AUC:
        report["eligible"] = True
        report["reason"] = f"{best} beats the rule-based baseline by {gain:.3f} AUC (more than one fold std) in grouped cross-validation."
    else:
        report["reason"] = (f"{best} does not clearly beat the rule-based baseline (needs gain >= {C.MIN_AUC_GAIN}, gain > fold std, "
                            f"and AUC >= {C.MIN_ABS_AUC}). Rules stay in charge.")
    if simulated:
        report["reason"] += " Trained on SIMULATED data: for pipeline testing only, never activated."
    elif report["eligible"]:
        C.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, C.MODEL_FILE)
        report["active"] = True
    return _save_report(report, simulated)


def _save_report(report: dict, simulated: bool) -> dict:
    path = C.DEMO_REPORT_FILE if simulated else C.MODEL_REPORT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not simulated and not report.get("active") and C.MODEL_FILE.exists():
        C.MODEL_FILE.unlink()          # a model that no longer qualifies must not stay active
    path.write_text(json.dumps(report, indent=2))
    return report


def load_report(simulated: bool = False) -> dict | None:
    path = C.DEMO_REPORT_FILE if simulated else C.MODEL_REPORT_FILE
    return json.loads(path.read_text()) if path.exists() else None


def load_active_model():
    rep = load_report(False)
    if rep and rep.get("active") and C.MODEL_FILE.exists():
        return joblib.load(C.MODEL_FILE)
    return None


def predict_like_probability(model, feats: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(feats[CAT + NUM])[:, 1]
