"""Central configuration: labels, allowed values, weights and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXERCISE_FILE = ROOT / "exercises.csv"
FEEDBACK_FILE = ROOT / "data" / "feedback.csv"
SIMULATED_FEEDBACK_FILE = ROOT / "data" / "feedback_SIMULATED.csv"
MODEL_FILE = ROOT / "models" / "feedback_model.joblib"
MODEL_REPORT_FILE = ROOT / "models" / "model_report.json"
DEMO_REPORT_FILE = ROOT / "models" / "demo_report.json"

LEVELS = ["beginner", "intermediate", "advanced"]
DIFFICULTIES = ["low", "moderate", "high"]
MUSCLE_GROUPS = ["chest", "back", "shoulders", "biceps", "triceps", "legs", "core"]
EQUIPMENT = ["barbell", "dumbbells", "cable_machine", "resistance_bands", "machines", "pull_up_bar", "bodyweight"]
EXERCISE_TYPES = ["compound", "isolation", "isometric", "stability"]
LOCATIONS = ["gym", "home"]
REP_UNITS = ["reps", "seconds"]
INTENSITIES = ["low", "moderate", "high"]
DURATIONS = [30, 45, 60, 75, 90]

GOALS = {
    "muscle_gain": ("Build muscle", "Moderate loads, 8-12 reps"),
    "strength": ("Get stronger", "Heavier loads, fewer reps"),
    "general_fitness": ("General fitness", "Balanced and sustainable"),
    "fat_loss": ("Lose fat", "Higher reps, shorter rests"),
}
LEVEL_INFO = {
    "beginner": "New to structured training or unfamiliar with many exercises",
    "intermediate": "Consistent, and comfortable with the common exercises",
    "advanced": "Experienced with structured training and variations",
}
MUSCLE_INFO = {
    "chest": "Upper, middle and lower emphasis",
    "back": "Lats, upper back, mid-back, lower back",
    "shoulders": "Front, lateral and rear delts",
    "biceps": "Elbow flexors, long and short head",
    "triceps": "Long, lateral and medial heads",
    "legs": "Quads, hamstrings, glutes, calves",
    "core": "Abs, obliques, trunk stabilisers",
}
EQUIPMENT_LABELS = {
    "barbell": "Barbell", "dumbbells": "Dumbbells", "cable_machine": "Cable machine",
    "resistance_bands": "Resistance bands", "machines": "Machines",
    "pull_up_bar": "Pull-up bar", "bodyweight": "Bodyweight",
}
PROTECT_AREAS = {"shoulder": ["shoulder"], "lower_back": ["lower back"], "wrist": ["wrist"]}

# Weighted ranking (spec section 6.2). Starting weights, not scientifically validated.
WEIGHTS = {"muscle": 0.35, "equipment": 0.20, "level": 0.20, "goal": 0.15, "preference": 0.10}
ML_BLEND = 0.20            # share of the final score a *validated* feedback model may take
MMR_LAMBDA = 0.75          # relevance vs diversity when composing a session
WARMUP_MINUTES = 6
TRANSITION_MINUTES = 1.0
MAX_EXERCISES = 8
MIN_FEEDBACK_ROWS = 100
MIN_CLASS_ROWS = 20
MIN_AUC_GAIN = 0.02
MIN_ABS_AUC = 0.60
