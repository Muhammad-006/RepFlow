"""
exercises/base.py — the contract every RepFlow coach implements.

A coach turns a stream of pose keypoints into three things: a count, a
live picture of what the body is doing, and a *ranked list of things
worth saying*. It never speaks — it proposes. The session layer
(session.py) decides what actually gets voiced, because only it knows how
long ago the last cue played.

Two rules in here exist specifically to stop the app talking nonsense:

`MotionGate`
    If the tracked angle has barely moved for the last second, the athlete
    is holding a position, not repping. A held position must never advance
    a rep state machine and must never trigger a movement correction. This
    is what used to produce "lower your chest" while someone was sitting
    still in a plank: pose noise of a few degrees, frame after frame,
    eventually crossed a threshold and the machine believed a rep had
    started and been abandoned.

`min_partial_rom`
    A rep attempt is only worth commenting on if it actually went
    somewhere. Coming back to the top after 8° of travel is measurement
    noise; coming back after 30° is a genuinely shallow rep and deserves a
    word.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from biomech import DESIGN_DT_S, Keypoints, Smoother, clamp


@dataclass
class RepRecord:
    """One completed rep (or, for a hold, one scored block of time)."""
    index: int
    t: float                      # seconds since set start
    duration: float
    score: int                    # 0-100
    clean: bool
    fault: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Tick:
    """Everything one frame of coaching produced."""
    count: int = 0                       # reps completed, or whole seconds held
    stage: str = "SETUP"
    is_calibrated: bool = False
    calibration_pct: float = 0.0
    primary: Optional[float] = None      # the main tracked angle
    secondary: Optional[float] = None    # the supporting angle
    faults: Tuple[str, ...] = ()
    cues: Tuple[str, ...] = ()           # candidates, best first
    record: Optional[RepRecord] = None
    quality: Optional[int] = None        # live 0-100 for this moment
    hold_s: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class MotionGate:
    """Is the athlete moving, or holding still?

    Judged over a trailing TIME window, not a fixed number of received
    frames. A fixed sample-COUNT window silently changes meaning with the
    achieved frame rate: 10 samples is one second at the design cadence
    (`DESIGN_DT_S`) but ten seconds once the network round trip has
    degraded toward ~1 fps — and ten seconds is long enough that a real
    athlete has usually started and finished an entire rep before
    "moving" would even trigger, or before calibration (see `_calibrate`
    below, which uses this same gate) would ever complete.

    `window` is still accepted as a frame count for anyone constructing
    this with the old signature; it is converted to seconds against
    `DESIGN_DT_S` so behaviour at the design cadence is unchanged.

    MIN_KEEP is load-bearing, and its absence was a real bug. Trimming
    purely by age kept only TWO samples the moment the gap between frames
    reached half the window (~0.5 s, i.e. 2 fps), because each new sample
    pushed the one before last outside the cutoff. `is_still` needs three.
    So below roughly 1.5 fps `is_still` could never be True no matter how
    motionless the athlete was — and since all three coaches gate
    calibration on `is_still`, calibration never completed, and a set that
    never calibrates never counts a rep or a single second of a hold. That
    is the whole of "it detected me but the counter stayed on zero". The
    window is a *maximum* age, never a reason to hold fewer samples than
    the measurement needs.
    """

    MIN_KEEP = 3

    def __init__(self, window: int = 10, still_deg: float = 4.0, window_s: Optional[float] = None):
        self.window_s = window_s if window_s is not None else window * DESIGN_DT_S
        self.still_deg = still_deg
        self._buf: List[Tuple[float, float]] = []   # (timestamp, value)
        self._synthetic_t = 0.0   # used only when no timestamp is ever given

    def reset(self):
        self._buf.clear()
        self._synthetic_t = 0.0

    def push(self, value: Optional[float], now: Optional[float] = None) -> None:
        if value is None:
            return
        if now is None:
            # No timing info available (e.g. a caller not yet updated) —
            # fall back to counting frames at the design cadence, which
            # reproduces the old frame-count behaviour exactly.
            self._synthetic_t += DESIGN_DT_S
            now = self._synthetic_t
        self._buf.append((now, value))
        cutoff = now - self.window_s
        while len(self._buf) > self.MIN_KEEP and self._buf[0][0] < cutoff:
            self._buf.pop(0)
        if len(self._buf) > 400:   # hard cap: never grow unbounded on bad timestamps
            del self._buf[:-400]

    @property
    def samples(self) -> List[float]:
        """Plain values, oldest first — kept for callers (pushup.py,
        squat.py) that seed a rep's starting bound from recent history."""
        return [v for _, v in self._buf]

    @property
    def spread(self) -> float:
        vals = self.samples
        if len(vals) < 2:
            return 0.0
        return max(vals) - min(vals)

    @property
    def is_still(self) -> bool:
        if len(self._buf) < self.MIN_KEEP:
            return False
        span = self._buf[-1][0] - self._buf[0][0]
        # Enough real time observed to be worth an opinion. At the design
        # cadence this is the old half-window rule; at a degraded frame
        # rate MIN_KEEP samples already span far more than half a window,
        # so this passes on the strength of elapsed time rather than of
        # sample count — which is the point.
        return span >= self.window_s * 0.5 and self.spread < self.still_deg

    @property
    def is_moving(self) -> bool:
        return self.spread >= self.still_deg


class ExerciseCoach:
    # --- identity -----------------------------------------------------
    KEY: str = ""
    NAME: str = ""
    UNIT: str = "reps"              # "reps" | "seconds"
    PRIMARY_LABEL: str = ""
    SECONDARY_LABEL: str = ""
    SETUP_HINT: str = ""
    CAMERA_VIEW: str = "side"       # "side" | "front" | "either"

    # --- tuning knobs a subclass overrides ----------------------------
    smoothing_alpha: float = 0.5
    max_jump_deg: float = 45.0
    still_deg: float = 4.0
    min_partial_rom: float = 22.0
    # How long a steady reading has to be HELD, in real seconds, before
    # calibration locks in. Was `calibration_frames = 8` (a frame count,
    # implicitly ~0.8 s at the design cadence) — expressed in seconds now
    # so it means the same thing regardless of the frame rate actually
    # achieved. 8 frames * DESIGN_DT_S kept as the default so behaviour at
    # the design cadence is unchanged.
    calibration_seconds: float = 8 * DESIGN_DT_S

    def __init__(self):
        self.smoother = Smoother(self.smoothing_alpha, self.max_jump_deg)
        self.motion = MotionGate(still_deg=self.still_deg)
        self.records: List[RepRecord] = []
        self.set_start = 0.0
        self.reset(0.0)

    # ------------------------------------------------------------------
    def reset(self, now: float) -> None:
        self.smoother.reset()
        self.motion.reset()
        self.records = []
        self.set_start = now
        self.count = 0
        self.is_calibrated = False
        self._calib: List[float] = []
        self._calib_start: Optional[float] = None
        self._calib_elapsed = 0.0
        self._reset_state(now)

    def _reset_state(self, now: float) -> None:
        """Subclass hook — clear exercise-specific state."""

    # ------------------------------------------------------------------
    def update(self, kp: Optional[Keypoints], now: float, detected: bool) -> Tick:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def summary(self, elapsed: float) -> Dict:
        """Default summary shape. Subclasses add their own numbers."""
        recs = self.records
        n = len(recs)
        if n == 0:
            return {
                "count": 0, "unit": self.UNIT, "clean": 0, "clean_pct": 0,
                "form_score": 0, "grade": "—", "duration_s": round(elapsed, 1),
                "rep_list": [], "faults": {}, "insights": [self.empty_insight()],
            }
        scores = [r.score for r in recs]
        avg = statistics.mean(scores)
        clean = sum(1 for r in recs if r.clean)
        faults: Dict[str, int] = {}
        for r in recs:
            if r.fault:
                faults[r.fault] = faults.get(r.fault, 0) + 1
        consistency = max(0, round(100 - statistics.pstdev(scores) * 2.5)) if n > 1 else 100
        return {
            "count": self.count, "unit": self.UNIT,
            "clean": clean, "clean_pct": round(100 * clean / n),
            "form_score": round(avg), "grade": grade_for(avg),
            "consistency": consistency,
            "duration_s": round(elapsed, 1),
            "rep_list": [rec_dict(r) for r in recs],
            "faults": faults,
            "insights": self.insights(recs, elapsed),
        }

    def insights(self, recs: List[RepRecord], elapsed: float) -> List[str]:
        return []

    def empty_insight(self) -> str:
        return "Nothing was counted. " + self.SETUP_HINT

    # ------------------------------------------------------------------
    def _calibrate(self, value: float, accept_min: float, accept_max: float,
                   now: Optional[float] = None) -> bool:
        """Collect a run of frames in a plausible 'ready position' range,
        held for `calibration_seconds` of real time — not a fixed number
        of frames. At the design frame rate these are the same thing; at
        a much lower achieved rate (a slow network round trip), waiting
        for a fixed frame COUNT means waiting far longer, in actual
        seconds, than any real athlete will patiently hold a plank before
        starting their set — which is exactly what left calibration (and
        therefore every rep after it) never completing.

        `now` is optional for backward compatibility with any direct
        caller that doesn't have a timestamp handy; it then falls back to
        counting frames at the design cadence, reproducing the old
        behaviour exactly.
        """
        if now is None:
            now = (self._calib_start or 0.0) + len(self._calib) * DESIGN_DT_S
        if accept_min <= value <= accept_max:
            if not self._calib:
                self._calib_start = now
            self._calib.append(value)
            self._calib_elapsed = now - self._calib_start
        elif self._calib:
            self._calib.clear()      # movement during calibration — start over
            self._calib_elapsed = 0.0
        if self._calib and self._calib_elapsed >= self.calibration_seconds:
            self.on_calibrated(statistics.mean(self._calib))
            self._calib.clear()
            self._calib_elapsed = 0.0
            self.is_calibrated = True
            return True
        return False

    @property
    def calibration_pct(self) -> float:
        if self.is_calibrated:
            return 1.0
        return clamp(self._calib_elapsed / self.calibration_seconds, 0.0, 1.0)

    def on_calibrated(self, baseline: float) -> None:
        """Subclass hook — set thresholds from the observed ready position."""


# ----------------------------------------------------------------------
def grade_for(score: float) -> str:
    for g, cut in (("A+", 95), ("A", 88), ("B+", 82), ("B", 75), ("C", 65), ("D", 0)):
        if score >= cut:
            return g
    return "D"


def rec_dict(r: RepRecord) -> Dict:
    d = {"rep": r.index, "t": round(r.t, 2), "duration": round(r.duration, 2),
         "score": r.score, "clean": r.clean, "fault": r.fault}
    d.update({k: round(v, 1) for k, v in r.metrics.items()})
    return d


def score_parts(parts: Dict[str, float], weights: Dict[str, float]) -> int:
    total = sum(parts[k] * w for k, w in weights.items())
    return int(round(clamp(total, 0, 100)))
