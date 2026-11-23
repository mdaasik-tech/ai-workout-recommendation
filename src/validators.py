"""User profile model and input validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from . import config as C


@dataclass
class UserProfile:
    goal: str = ""
    level: str = ""
    muscle_group: str = ""
    emphasis: list = field(default_factory=list)      # optional muscle_emphasis values
    location: str = "gym"
    equipment: list = field(default_factory=list)
    duration: int = 45
    intensity: str = "moderate"
    pain: bool = False
    injury: bool = False
    medical: bool = False
    protect_areas: list = field(default_factory=list)  # shoulder / lower_back / wrist
    avoid_text: str = ""
    prefer_text: str = ""
    name: str = ""
    age: int | None = None


def validate_profile(p: UserProfile) -> list[str]:
    errors = []
    if p.goal not in C.GOALS: errors.append("Choose a fitness goal.")
    if p.level not in C.LEVELS: errors.append("Choose an experience level.")
    if p.muscle_group not in C.MUSCLE_GROUPS: errors.append("Choose a muscle group.")
    if p.location not in C.LOCATIONS: errors.append("Choose gym or home.")
    if not p.equipment: errors.append("Select at least one piece of equipment (or bodyweight).")
    elif set(p.equipment) - set(C.EQUIPMENT): errors.append("Unknown equipment selected.")
    if p.duration not in C.DURATIONS: errors.append("Choose a session duration.")
    if p.intensity not in C.INTENSITIES: errors.append("Choose an intensity.")
    if p.age is not None and not (13 <= p.age <= 90): errors.append("Age must be between 13 and 90.")
    return errors


def tokens(text: str) -> list[str]:
    """Comma/newline separated free text -> lowercase tokens of 3+ characters."""
    return [t.strip().lower() for t in str(text or "").replace("\n", ",").split(",") if len(t.strip()) >= 3]
