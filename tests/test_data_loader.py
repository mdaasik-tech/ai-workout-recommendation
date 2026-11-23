import pandas as pd
import pytest
from src import config as C
from src.data_loader import DataValidationError, load_exercises, validate_exercises


def test_loads_real_catalogue():
    df = load_exercises()
    assert len(df) > 100 and df.exercise_id.is_unique
    assert set(df.muscle_group) == set(C.MUSCLE_GROUPS)


def test_missing_column_gives_clear_error(tmp_path):
    bad = pd.read_csv(C.EXERCISE_FILE).drop(columns=["equipment"])
    f = tmp_path / "bad.csv"; bad.to_csv(f, index=False)
    with pytest.raises(DataValidationError, match="Missing columns"):
        load_exercises(f)


def test_unknown_alternative_and_value_detected():
    df = pd.read_csv(C.EXERCISE_FILE)
    df.loc[0, "alternative_ids"] = "EX999"
    df.loc[1, "equipment"] = "Dumbbell"
    problems = " ".join(validate_exercises(df))
    assert "EX999" in problems and "equipment" in problems
