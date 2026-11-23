import numpy as np
import pandas as pd
import pytest
from src import config as C
from src.data_loader import load_exercises
from src.ml_ranker import CAT, FEEDBACK_COLUMNS, NUM, ContentSimilarity, load_active_model, train_feedback_model

DF = load_exercises()


def test_similarity_is_symmetric_and_finds_same_group():
    sim = ContentSimilarity(DF)
    a, b = DF.exercise_id[0], DF.exercise_id[1]
    assert sim.sim(a, b) == pytest.approx(sim.sim(b, a))
    top = sim.similar(a, 5)
    assert len(top) == 5 and top[0][1] >= top[-1][1]
    assert DF.set_index("exercise_id").loc[top[0][0], "muscle_group"] == DF.loc[0, "muscle_group"]


def _fake(n_sessions, signal, rng):
    rows = []
    for s in range(n_sessions):
        for _ in range(5):
            r = DF.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
            lvl = rng.choice(C.LEVELS)
            rule = float(rng.uniform(0.6, 1.0))
            hidden = (r.exercise_type == "isolation") * 1.0 if signal else 0.0     # something the rules do not know
            z = 2.5 + 0.0 * rule + 2.0 * hidden + rng.normal(0, 0.6)
            rows.append(dict(timestamp="t", session_id=f"s{s}", exercise_id=r.exercise_id, duration=45, user_level=lvl, goal="muscle_gain",
                             intensity="moderate", muscle_group=r.muscle_group, movement_pattern=r.movement_pattern, equipment=r.equipment,
                             exercise_type=r.exercise_type, difficulty=r.difficulty, level_gap=1, rule_score=rule,
                             completion="completed", perceived_difficulty="appropriate", rating=int(np.clip(round(z), 1, 5)), is_simulated=False))
    return pd.DataFrame(rows)[FEEDBACK_COLUMNS]


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODEL_FILE", tmp_path / "m.joblib")
    monkeypatch.setattr(C, "MODEL_REPORT_FILE", tmp_path / "r.json")
    monkeypatch.setattr(C, "DEMO_REPORT_FILE", tmp_path / "d.json")


def test_too_little_data_is_refused(tmp_paths):
    rep = train_feedback_model(_fake(5, True, np.random.default_rng(0)))
    assert not rep["eligible"] and not rep["active"] and "Need at least" in rep["reason"]
    assert load_active_model() is None


def test_model_activates_only_when_it_beats_baseline(tmp_paths):
    rep = train_feedback_model(_fake(120, True, np.random.default_rng(1)))
    assert rep["eligible"] and rep["active"] and load_active_model() is not None


def test_noise_data_does_not_activate(tmp_paths):
    rep = train_feedback_model(_fake(120, False, np.random.default_rng(2)))
    assert not rep["active"]


def test_simulated_data_never_activates(tmp_paths):
    rep = train_feedback_model(_fake(120, True, np.random.default_rng(3)), simulated=True)
    assert rep["eligible"] and not rep["active"] and load_active_model() is None
