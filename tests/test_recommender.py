import pytest
from src import config as C
from src.data_loader import load_exercises
from src.ml_ranker import ContentSimilarity
from src.validators import UserProfile, validate_profile
from src.workout_builder import build_workout, get_alternatives, replace_exercise

DF = load_exercises()
SIM = ContentSimilarity(DF)


def prof(**kw):
    base = dict(goal="muscle_gain", level="intermediate", muscle_group="back", location="gym",
                equipment=["cable_machine", "dumbbells", "pull_up_bar"], duration=45, intensity="moderate")
    base.update(kw)
    return UserProfile(**base)


def names(w): return [i.row["exercise_name"] for i in w.items]


def test_beginner_chest_dumbbells_only_matches():
    w = build_workout(DF, prof(level="beginner", muscle_group="chest", equipment=["dumbbells"]), SIM)
    assert w.items and all(i.row["equipment"] == "dumbbells" and "beginner" in i.row["level_list"] for i in w.items)


def test_home_bodyweight_has_no_machine_or_barbell():
    w = build_workout(DF, prof(location="home", equipment=["bodyweight"], muscle_group="legs"), SIM)
    assert w.items and all(i.row["equipment"] == "bodyweight" and "home" in i.row["location_list"] for i in w.items)


def test_no_match_returns_empty_plan_not_forced_result():
    w = build_workout(DF, prof(muscle_group="biceps", equipment=["barbell"], location="home"), SIM)
    assert w.items == []


def test_avoided_exercise_is_excluded():
    base = build_workout(DF, prof(muscle_group="chest", equipment=["bodyweight", "dumbbells"]), SIM)
    target = base.items[0].row["exercise_name"].split()[-1].lower()
    again = build_workout(DF, prof(muscle_group="chest", equipment=["bodyweight", "dumbbells"], avoid_text=target), SIM)
    assert all(target not in n.lower() for n in names(again))


def test_protected_area_removes_cautioned_exercises():
    w = build_workout(DF, prof(muscle_group="shoulders", equipment=C.EQUIPMENT, protect_areas=["shoulder"]), SIM)
    assert all("shoulder" not in i.row["contraindication_notes"].lower() for i in w.items)


def test_pain_flag_removes_high_difficulty_and_warns():
    w = build_workout(DF, prof(muscle_group="legs", equipment=C.EQUIPMENT, level="advanced", pain=True), SIM)
    assert all(i.row["difficulty"] != "high" for i in w.items) and any("physiotherapist" in m for m in w.messages)


@pytest.mark.parametrize("dur", C.DURATIONS)
def test_plan_fits_duration(dur):
    w = build_workout(DF, prof(muscle_group="legs", equipment=C.EQUIPMENT, duration=dur), SIM)
    assert w.est_minutes <= dur + 1 and len(w.items) >= 2


def test_session_is_varied():
    w = build_workout(DF, prof(muscle_group="legs", equipment=C.EQUIPMENT, duration=90), SIM)
    pats = [i.row["movement_pattern"] for i in w.items]
    assert len(set(pats)) >= len(pats) - 2


def test_alternatives_respect_constraints_and_replace_works():
    p = prof(muscle_group="chest", equipment=["dumbbells", "bodyweight", "resistance_bands"], level="beginner", duration=30)
    w = build_workout(DF, p, SIM)
    alts = get_alternatives(DF, p, SIM, w, 0)
    assert alts
    used = {i.row["exercise_id"] for i in w.items}
    for a in alts:
        row = DF[DF.exercise_id == a["id"]].iloc[0]
        assert a["id"] not in used and row.equipment in p.equipment and "beginner" in row.level_list
    w2 = replace_exercise(DF, p, w, 0, alts[0]["id"])
    assert w2.items[0].row["exercise_id"] == alts[0]["id"]


def test_replace_rejects_disallowed_exercise():
    p = prof(muscle_group="chest", equipment=["dumbbells"], level="beginner")
    w = build_workout(DF, p, SIM)
    barbell = DF[(DF.muscle_group == "chest") & (DF.equipment == "barbell")].exercise_id.iloc[0]
    assert replace_exercise(DF, p, w, 0, barbell).items[0].row["exercise_id"] == w.items[0].row["exercise_id"]


def test_profile_validation():
    assert validate_profile(prof(age=5)) and validate_profile(prof(equipment=[])) and not validate_profile(prof(age=30))
