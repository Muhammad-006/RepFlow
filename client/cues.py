"""
cues.py (phone) — cue_id → what to show on screen and what to play.

Keys must stay identical to `server/cues.py`. The server only ever sends
an id; the phone owns the words and the audio, which is why the coaching
voice has no network latency at all and why the wire carries no English.

`TONE` drives the colour of the on-screen banner: "good" is green,
"tip" is amber, "warn" is red.
"""

from __future__ import annotations

# cue_id -> (text on screen, audio file or None, tone)
CUES: dict[str, tuple[str, str | None, str]] = {
    # --- shared -------------------------------------------------------
    "calibrating":   ("Hold still a second — calibrating…",   "audio/calibrating.wav",   "tip"),
    "lost_tracking": ("Can't see you — step back into frame", "audio/lost_tracking.wav", "warn"),
    "rep_good":      ("Nice, that's clean",                   "audio/rep_good.wav",      "good"),
    "tempo_cue":     ("Slow it down, stay in control",        "audio/tempo_cue.wav",     "tip"),
    "goal_reached":  ("That's your goal — nice work",         "audio/goal_reached.wav",  "good"),
    "set_complete":  ("Set complete. Well done",              "audio/set_complete.wav",  "good"),

    # --- push-up ------------------------------------------------------
    "pu_depth":    ("Go a little deeper next one",         "audio/pu_depth.wav",    "tip"),
    "pu_hip_sag":  ("Hips dropping — squeeze your glutes", "audio/pu_hip_sag.wav",  "warn"),
    "pu_hip_pike": ("Hips too high, flatten out",          "audio/pu_hip_pike.wav", "tip"),
    "pu_lockout":  ("Push all the way up at the top",      "audio/pu_lockout.wav",  "tip"),

    # --- squat --------------------------------------------------------
    "sq_depth": ("A bit lower next rep",  "audio/sq_depth.wav", "tip"),
    "sq_chest": ("Chest up, stay tall",   "audio/sq_chest.wav", "tip"),
    "sq_knees": ("Push your knees out",   "audio/sq_knees.wav", "warn"),
    "sq_stand": ("Stand all the way up",  "audio/sq_stand.wav", "tip"),

    # --- plank --------------------------------------------------------
    "pl_sag":    ("Lift your hips back up",   "audio/pl_sag.wav",    "warn"),
    "pl_pike":   ("Drop your hips a touch",   "audio/pl_pike.wav",   "tip"),
    "pl_good":   ("Great line — hold it",     "audio/pl_good.wav",   "good"),
    "pl_broken": ("Get back into position",   "audio/pl_broken.wav", "warn"),
    "pl_half":   ("Halfway — stay tight",     "audio/pl_half.wav",   "good"),
    "pl_30":     ("Thirty seconds",           "audio/pl_30.wav",     "good"),
    "pl_60":     ("One minute — nice",        "audio/pl_60.wav",     "good"),
}

# Short non-speech sounds, for rhythm rather than instruction.
EARCONS: dict[str, str] = {
    "tick": "audio/rep_tick.wav",
    "goal": "audio/goal.wav",
}


def text_of(cue_id: str) -> str:
    entry = CUES.get(cue_id)
    return entry[0] if entry else ""


def tone_of(cue_id: str) -> str:
    entry = CUES.get(cue_id)
    return entry[2] if entry else "tip"
