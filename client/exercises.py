"""
exercises.py (phone) — how each movement is presented in the app.

The server is the source of truth for *what exists* (it sends its catalog
in the handshake). This file is the source of truth for what each one
looks like: icon, colour, the setup line, the goal options and what the
big number on the live screen means.

Keeping these apart is what makes adding a movement a small job — the
app picks up a new coach from the catalog and falls back to sensible
presentation for anything it doesn't have artwork for yet.
"""

from __future__ import annotations

import flet as ft

# key -> presentation
PRESENTATION: dict[str, dict] = {
    "pushup": {
        "name": "Push-Up",
        "tagline": "Depth, hip line, lockout and tempo",
        "icon": ft.Icons.FITNESS_CENTER,
        "unit": "reps",
        "camera_view": "side",
        "goals": [(5, "5"), (10, "10"), (15, "15"), (20, "20"), (0, "∞")],
        "goal_label": "rep goal",
        "setup": [
            (ft.Icons.PHONE_ANDROID, "Phone at hip height, about 2 m away"),
            (ft.Icons.SWAP_HORIZ, "Side-on, whole body in frame"),
            (ft.Icons.VOLUME_UP_OUTLINED, "Volume up — the coach talks you through it"),
        ],
    },
    "squat": {
        "name": "Squat",
        "tagline": "Depth, knee tracking and chest position",
        "icon": ft.Icons.ACCESSIBILITY_NEW,
        "unit": "reps",
        # Matches the server's SquatCoach.CAMERA_VIEW: squats read fine
        # side-on or at 45°, so this isn't "side" like push-up/plank, but
        # it isn't a fixed front-on "standing" shot either. Anything other
        # than "side" draws the upright placement sketch (see
        # screens.placement_diagram), which is the right guide either way.
        "camera_view": "either",
        "goals": [(5, "5"), (10, "10"), (15, "15"), (20, "20"), (0, "∞")],
        "goal_label": "rep goal",
        "setup": [
            (ft.Icons.PHONE_ANDROID, "Phone at hip height, about 2 m away"),
            (ft.Icons.SWAP_HORIZ, "Side-on or at 45° — head and feet both in frame"),
            (ft.Icons.VOLUME_UP_OUTLINED, "Volume up — the coach talks you through it"),
        ],
    },
    "plank": {
        "name": "Plank",
        "tagline": "Body line, stability and honest hold time",
        "icon": ft.Icons.TIMER_OUTLINED,
        "unit": "seconds",
        "camera_view": "side",
        "goals": [(30, "30s"), (45, "45s"), (60, "60s"), (90, "90s"), (0, "∞")],
        "goal_label": "hold target",
        "setup": [
            (ft.Icons.PHONE_ANDROID, "Phone at hip height, about 2 m away"),
            (ft.Icons.SWAP_HORIZ, "Side-on, head and feet both in frame"),
            (ft.Icons.VOLUME_UP_OUTLINED, "Volume up — you'll get told when to fix the line"),
        ],
    },
}

FALLBACK = {
    "name": "Exercise",
    "tagline": "",
    "icon": ft.Icons.SPORTS_GYMNASTICS,
    "unit": "reps",
    "camera_view": "side",
    "goals": [(5, "5"), (10, "10"), (15, "15"), (20, "20"), (0, "∞")],
    "goal_label": "goal",
    "setup": [(ft.Icons.PHONE_ANDROID, "Whole body in frame, phone at hip height")],
}

ORDER = ["pushup", "squat", "plank"]

# key -> presentation for movements that aren't live yet. These are shown
# on the home screen as locked "coming soon" cards regardless of whether
# the server has advertised them in its roadmap list, so the teaser is
# never at the mercy of the handshake actually completing.
ROADMAP_PRESENTATION: dict[str, dict] = {
    "pullup": {"name": "Pull-Ups", "icon": ft.Icons.SPORTS_GYMNASTICS,
               "note": "Dead-hang depth and chin-over-bar tracking"},
    "dip": {"name": "Tricep Dips", "icon": ft.Icons.SPORTS_MARTIAL_ARTS,
            "note": "Elbow depth and lockout, bars or bench"},
    "lunge": {"name": "Lunges", "icon": ft.Icons.DIRECTIONS_WALK,
              "note": "Front-knee tracking and balance, left/right"},
}

ROADMAP_ORDER = ["pullup", "dip", "lunge"]


def presentation(key: str) -> dict:
    return PRESENTATION.get(key, FALLBACK)


def merge_catalog(catalog: list) -> list:
    """Server catalog + local presentation, in a sensible order."""
    by_key = {c.get("key"): c for c in (catalog or [])}
    if not by_key:
        by_key = {k: {"key": k, "name": v["name"], "unit": v["unit"]}
                  for k, v in PRESENTATION.items()}
    keys = [k for k in ORDER if k in by_key] + [k for k in by_key if k not in ORDER]
    out = []
    for k in keys:
        entry = dict(presentation(k))
        entry.update({kk: vv for kk, vv in by_key[k].items() if vv})
        entry["key"] = k
        out.append(entry)
    return out


def roadmap_catalog(roadmap: list) -> list:
    """Presentation for movements that aren't live yet, always returning the
    known three (Pull-Ups, Tricep Dips, Lunges) so the home screen can show
    them as locked cards even before a session has connected.

    Unlike merge_catalog, the server doesn't get the final word on name/icon
    here — these are just a teaser, and the curated names above are what we
    want on the card. A server-side roadmap note fills in the "why" line
    only when we don't already have one locally; an entirely unknown key
    the server adds later still shows up, using whatever name it sent."""
    by_key = {r.get("key"): r for r in (roadmap or [])}
    keys = list(ROADMAP_ORDER) + [k for k in by_key if k not in ROADMAP_ORDER]
    out = []
    for k in keys:
        server = by_key.get(k) or {}
        if k in ROADMAP_PRESENTATION:
            entry = dict(ROADMAP_PRESENTATION[k])
        else:
            entry = {"name": server.get("name", k.title()), "icon": ft.Icons.SPORTS_GYMNASTICS,
                      "note": server.get("note", "In development for a future version")}
        entry["key"] = k
        out.append(entry)
    return out


def unit_label(unit: str, count: int) -> str:
    if unit == "seconds":
        return "SECONDS"
    return "REPS"


def format_count(unit: str, count: int) -> str:
    if unit == "seconds":
        return f"{int(count) // 60}:{int(count) % 60:02d}" if count >= 60 else str(int(count))
    return str(int(count))
