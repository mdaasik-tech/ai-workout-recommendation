"""Headless UI tests using Streamlit's AppTest."""
from streamlit.testing.v1 import AppTest

from pathlib import Path
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def click(at, key):
    at.button(key=key).click().run()


def test_home_renders():
    at = run()
    assert not at.exception
    assert any("planned in seconds" in m.value for m in at.markdown)


def test_full_flow_generates_workout():
    at = run()
    click(at, "cta")
    click(at, "card_goal_muscle_gain"); click(at, "next")
    click(at, "card_level_intermediate"); click(at, "next")
    click(at, "card_muscle_back"); click(at, "next")
    click(at, "next")                       # gear & time with defaults
    assert at.session_state["step"] == 5
    click(at, "gen")
    assert not at.exception, at.exception
    assert at.session_state["page"] == "workout"
    w = at.session_state["workout"]
    assert w.items and all(i.row["muscle_group"] == "back" for i in w.items)
    assert any("Swap" in b.label for b in at.button)


def test_swap_changes_exercise():
    at = run()
    click(at, "cta")
    for k in ("card_goal_strength", "next", "card_level_beginner", "next", "card_muscle_chest", "next", "next"):
        click(at, k)
    click(at, "gen")
    before = [i.row["exercise_id"] for i in at.session_state["workout"].items]
    swap_btn = next(b for b in at.button if b.key and b.key.startswith("swap_0_"))
    swap_btn.click().run()
    after = [i.row["exercise_id"] for i in at.session_state["workout"].items]
    assert not at.exception and after[0] != before[0] and after[1:] == before[1:]


def test_no_match_message():
    at = run()
    click(at, "cta")
    at.session_state["ans"].update(goal="muscle_gain", level="beginner", muscle="biceps", loc="home", eq=["barbell"], areas=[])
    at.session_state["step"] = 5
    at.run()
    click(at, "gen")
    assert not at.exception
    assert any("No suitable match found" in m.value for m in at.markdown)


def test_ml_lab_and_about_render():
    at = run()
    click(at, "nav_ml")
    assert not at.exception, at.exception
    click(at, "nav_about")
    assert not at.exception
