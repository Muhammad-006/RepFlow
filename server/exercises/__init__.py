"""
exercises — the coach registry.

Adding a movement to RepFlow is: write a class that implements
`ExerciseCoach`, and add it to `REGISTRY` below. Nothing in the server,
the transport or the app needs to know the difference — the app asks
`/v1/catalog` what exists and builds its menu from the answer.
"""

from exercises.base import ExerciseCoach, RepRecord, Tick, grade_for
from exercises.plank import PlankCoach
from exercises.pushup import PushUpCoach
from exercises.squat import SquatCoach

REGISTRY = {c.KEY: c for c in (PushUpCoach, SquatCoach, PlankCoach)}
DEFAULT = PushUpCoach.KEY

# Movements that are designed but not shipped yet. The app shows these as
# a roadmap list rather than as fake locked buttons.
ROADMAP = [
    {"key": "pullup", "name": "Pull-Up", "note": "Chin clearance and kipping detection"},
    {"key": "dip", "name": "Dip", "note": "Shoulder depth and forward-lean control"},
    {"key": "lunge", "name": "Lunge", "note": "Front-knee tracking and hip level"},
]


def make(key: str) -> ExerciseCoach:
    return REGISTRY.get(key, REGISTRY[DEFAULT])()


def catalog() -> list:
    return [
        {
            "key": c.KEY,
            "name": c.NAME,
            "unit": c.UNIT,
            "primary_label": c.PRIMARY_LABEL,
            "secondary_label": c.SECONDARY_LABEL,
            "setup_hint": c.SETUP_HINT,
            "camera_view": c.CAMERA_VIEW,
        }
        for c in REGISTRY.values()
    ]


__all__ = ["REGISTRY", "DEFAULT", "ROADMAP", "make", "catalog",
           "ExerciseCoach", "RepRecord", "Tick", "grade_for"]
