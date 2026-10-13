"""
cues.py — the coach's entire vocabulary. Shared verbatim with the client.

The server decides *which* cue applies and sends only its id; the phone
plays the matching pre-rendered clip from `client/assets/audio/`. Nothing
but a short string ever crosses the wire for speech, which is why the
voice has no latency and works with the phone in a pocket.

`DURATION` is filled in by `tools/render_voice.py` when the clips are
rendered, and the throttler uses it so the coach never starts a second
sentence over the top of the first one. That was a real problem in v2:
cue clips ran 2-3 seconds against a 2.2 second minimum gap.

Writing rules for cue text, learned the hard way:
  * Address the *next* rep, not the last one. "A little deeper next time"
    lands; "that was shallow" is just criticism after the fact.
  * Under seven words. Anything longer is still playing when the athlete
    is already at the bottom of the next rep.
  * No numbers unless they're a milestone — "ninety degrees" means
    nothing to someone upside down in a push-up.
"""

from __future__ import annotations

import json
import os
from enum import IntEnum


class CuePriority(IntEnum):
    LOW = 1       # encouragement, status
    MEDIUM = 2    # technique corrections
    HIGH = 3      # safety, or "I've lost you"


# cue_id -> (spoken text, priority)
CUES: dict[str, tuple[str, CuePriority]] = {
    # --- shared -------------------------------------------------------
    "calibrating":   ("Hold still for a second, I'm calibrating.", CuePriority.LOW),
    "lost_tracking": ("I can't see you — step back into frame.",   CuePriority.HIGH),
    "rep_good":      ("Nice, that's clean.",                       CuePriority.LOW),
    "tempo_cue":     ("Slow it down, stay in control.",            CuePriority.MEDIUM),
    "goal_reached":  ("That's your goal. Nice work.",              CuePriority.LOW),
    "set_complete":  ("Set complete. Well done.",                  CuePriority.LOW),

    # --- push-up ------------------------------------------------------
    "pu_depth":     ("Go a little deeper next one.",          CuePriority.MEDIUM),
    "pu_hip_sag":   ("Hips are dropping — squeeze your glutes.", CuePriority.HIGH),
    "pu_hip_pike":  ("Hips too high, flatten out.",           CuePriority.MEDIUM),
    "pu_lockout":   ("Push all the way up at the top.",       CuePriority.MEDIUM),

    # --- squat --------------------------------------------------------
    "sq_depth":  ("A bit lower next rep.",           CuePriority.MEDIUM),
    "sq_chest":  ("Chest up, stay tall.",            CuePriority.MEDIUM),
    "sq_knees":  ("Push your knees out.",            CuePriority.HIGH),
    "sq_stand":  ("Stand all the way up.",           CuePriority.MEDIUM),

    # --- plank --------------------------------------------------------
    "pl_sag":    ("Lift your hips back up.",         CuePriority.HIGH),
    "pl_pike":   ("Drop your hips a touch.",         CuePriority.MEDIUM),
    "pl_good":   ("Great line — hold it there.",     CuePriority.LOW),
    "pl_broken": ("Get back into position.",         CuePriority.HIGH),
    "pl_half":   ("Halfway. Stay tight.",            CuePriority.LOW),
    "pl_30":     ("Thirty seconds.",                 CuePriority.LOW),
    "pl_60":     ("One minute. Nice.",               CuePriority.LOW),
}

# Non-speech sounds the phone plays for rhythm, not instruction.
EARCONS = ("rep_tick", "goal")

# Filled in at render time: cue_id -> clip length in seconds.
DURATION: dict[str, float] = {}

_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cue_manifest.json")
try:
    with open(_manifest, "r", encoding="utf-8") as _f:
        DURATION.update({k: float(v) for k, v in json.load(_f).get("duration", {}).items()})
except (OSError, ValueError):
    pass


def duration_of(cue_id: str, default: float = 1.6) -> float:
    return DURATION.get(cue_id, default)


def text_of(cue_id: str) -> str:
    entry = CUES.get(cue_id)
    return entry[0] if entry else ""
