"""
biomech.py — orientation-independent geometry shared by every exercise.

Everything here works on the normalized keypoint dict produced by
pose_engine (`{"left_elbow": (x, y, conf), ...}`) and deliberately avoids
assuming which way is "up" in the frame. The phone can be propped
portrait, landscape or upside-down; angles between three joints don't
care, and where we *do* need a sense of up (hip sag vs hip pike) we work
it out from the athlete's own contact points with the floor rather than
from the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

Keypoints = Dict[str, Tuple[float, float, float]]

MIN_CONF = 0.30


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------
def angle_abc(a, b, c) -> float:
    """Interior angle at b, in degrees, 0..180."""
    rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    ang = abs(math.degrees(rad))
    return 360.0 - ang if ang > 180.0 else ang


def side_of_line(p, a, b) -> float:
    """Signed area — which side of the line a→b the point p falls on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def conf_of(kp: Keypoints, names: Iterable[str]) -> float:
    vals = [kp.get(n, (0.0, 0.0, 0.0))[2] for n in names]
    return min(vals) if vals else 0.0


def mean_conf(kp: Keypoints, names: Iterable[str]) -> float:
    vals = [kp.get(n, (0.0, 0.0, 0.0))[2] for n in names]
    return sum(vals) / len(vals) if vals else 0.0


class SideTracker:
    """Picks the camera-facing side (left/right) and *sticks* to it.

    Flip-flopping between sides mid-rep was a real source of phantom angle
    jumps — one frame measuring the near arm, the next the far arm. We
    lock on to whichever side the model sees best and only switch after
    the locked side has been unusable for several consecutive frames.
    """

    def __init__(self, joints: Tuple[str, ...], min_conf: float = MIN_CONF, grace: int = 5):
        self.joints = joints
        self.min_conf = min_conf
        self.grace = grace
        self.locked: Optional[str] = None
        self._misses = 0

    def reset(self):
        self.locked = None
        self._misses = 0

    def pick(self, kp: Optional[Keypoints]) -> Tuple[Optional[str], float]:
        if not kp:
            return None, 0.0
        scores = {s: mean_conf(kp, [f"{s}_{j}" for j in self.joints]) for s in ("left", "right")}
        if self.locked and scores[self.locked] >= self.min_conf:
            self._misses = 0
            return self.locked, scores[self.locked]
        if self.locked:
            self._misses += 1
            if self._misses <= self.grace:
                return None, scores[self.locked]
        best = max(scores, key=scores.get)
        if scores[best] < self.min_conf:
            return None, scores[best]
        self.locked, self._misses = best, 0
        return best, scores[best]


DESIGN_DT_S = 0.1
"""The per-frame interval `Smoother`/`MotionGate`/calibration were tuned
against — one frame roughly every 100 ms, matching the client's own
MIN_SEND_INTERVAL_S. Every timing-sensitive constant in this module and in
exercises/base.py is expressed relative to this, specifically so that a
slow network round trip (frames arriving every 500 ms-1 s instead of every
100 ms) degrades tracking smoothly instead of silently breaking rep
counting outright. See Smoother.push and MotionGate below."""


class Smoother:
    """Exponential smoothing with an outlier gate, generalized to however
    much real time actually elapsed between samples.

    A single bad pose estimate used to be enough to trip a state change —
    that is most of what "it told me to go lower while I was holding a
    plank" actually was. Samples that jump more than `max_jump` degrees in
    one frame are treated as noise and damped hard instead of followed.

    Both `alpha` and `max_jump` were tuned assuming frames arrive roughly
    every `DESIGN_DT_S`. That assumption breaks the moment the network
    round trip gets slow: if a full push-up happens between two frames a
    full second apart instead of 100 ms apart, the real swing in elbow
    angle is far bigger than 45 deg — and without accounting for how much
    time actually passed, the fixed threshold treats that entire rep as
    noise and damps it to a quarter strength, again and again, so the
    smoothed angle the state machine sees barely moves even though the
    athlete completed the rep in full view of the camera. Scaling both by
    how many "design steps" the actual gap represents (`dt / DESIGN_DT_S`)
    fixes this: unchanged at the design cadence (the case every existing
    test exercises), and converging quickly to trust each new sample once
    frames are arriving seconds apart.

    (A velocity-adaptive version of this — speeding up during a sustained
    directional trend — was tried during an earlier fast-cadence bug fix,
    but it reopened the heavy-noise false-positive margin the fixed alpha
    was tuned against: a 300-frame random walk gets a same-direction run
    long enough to trip a "trend" often enough to matter. This is
    different: it reacts to the *measured* time gap, not an inferred
    trend, so it does nothing extra at a steady frame rate.)
    """

    DESIGN_DT_S = DESIGN_DT_S

    def __init__(self, alpha: float = 0.5, max_jump: float = 45.0):
        self.alpha = alpha
        self.max_jump = max_jump
        self.value: Optional[float] = None
        self._last_t: Optional[float] = None

    def reset(self):
        self.value = None
        self._last_t = None

    def push(self, sample: Optional[float], now: Optional[float] = None) -> Optional[float]:
        if sample is None:
            return self.value
        if self.value is None:
            self.value = sample
            self._last_t = now
            return self.value
        # `steps` is how many design-cadence frames the actual gap since
        # the last sample is worth. Floored at 1.0 deliberately: a burst
        # of frames arriving FASTER than design cadence must behave
        # exactly as before (this is the well-tested, steady-rate case),
        # and only a gap SLOWER than design cadence loosens the outlier
        # gate and speeds up convergence.
        steps = 1.0
        if now is not None and self._last_t is not None:
            dt = max(now - self._last_t, 0.0)
            steps = clamp(dt / self.DESIGN_DT_S, 1.0, 30.0)
        self._last_t = now
        delta = sample - self.value
        effective_max_jump = self.max_jump * steps
        a = self.alpha if abs(delta) <= effective_max_jump else self.alpha * 0.25
        # Closed form of applying the per-step alpha `steps` times over —
        # the standard way to generalize a fixed-interval EMA to whatever
        # interval actually occurred.
        a_eff = 1.0 - (1.0 - a) ** steps
        self.value += a_eff * delta
        return self.value


# ----------------------------------------------------------------------
# derived measurements
# ----------------------------------------------------------------------
@dataclass
class BodyLine:
    angle: Optional[float] = None          # shoulder–hip–ankle, 180 = straight
    deviation: Optional[float] = None      # signed: + sag toward floor, − pike up
    reliable: bool = False


def body_line(kp: Keypoints, side: str, floor_ref: str = "wrist",
              min_conf: float = MIN_CONF) -> BodyLine:
    """Shoulder–hip–ankle straightness, with the sign worked out from where
    the floor is.

    `floor_ref` is a joint we know is planted on the ground for this
    exercise (wrists in a push-up, ankles in a squat). Whichever side of
    the shoulder→ankle line that joint sits on is "down", so a hip on the
    same side is sagging and a hip on the other side is piked. No gravity
    vector needed, so it survives any camera rotation.
    """
    need = (f"{side}_shoulder", f"{side}_hip")
    if conf_of(kp, need) < min_conf:
        return BodyLine()

    def pt(name):
        x, y, _ = kp[f"{side}_{name}"]
        return (x, y)

    foot = "ankle" if kp.get(f"{side}_ankle", (0, 0, 0))[2] >= min_conf else "knee"
    if kp.get(f"{side}_{foot}", (0, 0, 0))[2] < min_conf:
        return BodyLine()
    if kp.get(f"{side}_{floor_ref}", (0, 0, 0))[2] < min_conf:
        # No floor reference — report straightness without a sign.
        ang = angle_abc(pt("shoulder"), pt("hip"), pt(foot))
        return BodyLine(angle=ang, deviation=None, reliable=True)

    shoulder, hip, ft_pt, ref = pt("shoulder"), pt("hip"), pt(foot), pt(floor_ref)
    ang = angle_abc(shoulder, hip, ft_pt)
    bend = 180.0 - ang
    sign = 1.0 if side_of_line(hip, shoulder, ft_pt) * side_of_line(ref, shoulder, ft_pt) > 0 else -1.0
    return BodyLine(angle=ang, deviation=sign * bend, reliable=True)


def joint_angle(kp: Keypoints, side: str, a: str, b: str, c: str,
                min_conf: float = MIN_CONF) -> Optional[float]:
    names = (f"{side}_{a}", f"{side}_{b}", f"{side}_{c}")
    if conf_of(kp, names) < min_conf:
        return None
    p = [kp[n][:2] for n in names]
    return angle_abc(p[0], p[1], p[2])


def hip_angle(kp: Keypoints, side: str, min_conf: float = MIN_CONF) -> Optional[float]:
    """Hip flexion: the angle at the hip between the torso and the thigh.

    180° standing tall, closing as you fold. This is the honest way to ask
    "is the chest up?" without any reference to which way is up in the
    image — it is measured entirely between the athlete's own segments, so
    it survives the phone being propped sideways.
    """
    return joint_angle(kp, side, "shoulder", "hip", "knee", min_conf)


def forward_fold(kp: Keypoints, side: str, min_conf: float = MIN_CONF) -> Optional[float]:
    """How much further the torso has folded than the legs have bent.

    A balanced squat closes the hip and the knee together, so this stays
    near zero however deep you go. A squat that turns into a good-morning
    closes the hip much further than the knee, and this grows.

    Being *more* upright than the legs (a high-bar or goblet squat, or
    anyone with good ankle mobility) is not a fault, so the measure floors
    at zero rather than firing a cue at people with good technique.
    """
    knee = joint_angle(kp, side, "hip", "knee", "ankle", min_conf)
    hip = hip_angle(kp, side, min_conf)
    if knee is None or hip is None:
        return None
    return max(0.0, knee - hip)


def knee_valgus(kp: Keypoints, min_conf: float = MIN_CONF) -> Optional[float]:
    """How far the knees have collapsed inward, as a fraction of hip width.

    Only meaningful from a front/three-quarter view, where both knees and
    both hips are visible. Returns None otherwise, so a side-on squat
    simply never raises a knee-cave fault instead of raising a wrong one.
    """
    need = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
    if conf_of(kp, need) < min_conf:
        return None
    hip_w = dist(kp["left_hip"][:2], kp["right_hip"][:2])
    knee_w = dist(kp["left_knee"][:2], kp["right_knee"][:2])
    ankle_w = dist(kp["left_ankle"][:2], kp["right_ankle"][:2])
    if hip_w < 0.02 or ankle_w < 0.02:
        return None
    # Side-on: hips and knees project almost on top of each other.
    if hip_w < 0.35 * max(ankle_w, 1e-6) and hip_w < 0.05:
        return None
    expected = max(hip_w, ankle_w)
    return max(0.0, (expected - knee_w) / expected)


def heel_lift(kp: Keypoints, side: str, min_conf: float = MIN_CONF) -> Optional[float]:
    """Positive when the ankle has travelled toward the toes relative to
    the knee — the signature of the heel coming off the floor."""
    names = (f"{side}_knee", f"{side}_ankle")
    if conf_of(kp, names) < min_conf:
        return None
    return None  # needs foot keypoints COCO-17 does not provide


class Stability:
    """Rolling wobble measure for holds (plank).

    Standard deviation of the body-line angle over a short window. A rock
    solid plank sits near zero; a hip that is slowly dropping, or a body
    that is shaking, climbs.

    The window is real time, not a frame count. `window=25` meant "about
    2.5 seconds" at the design cadence, but at one frame per second the
    same 25 frames spanned 25 seconds — long enough that the reading
    described the whole set rather than the current moment, so a plank
    that started clean and sagged late still read as rock solid. It also
    needed five samples before saying anything at all, which at that rate
    is five seconds of silence at the start of every hold. Both are now
    expressed in seconds, and the sample floor is three.

    `window` is still accepted as a frame count for backward
    compatibility and converted against DESIGN_DT_S, so behaviour at the
    design cadence is unchanged.
    """

    MIN_KEEP = 3

    def __init__(self, window: int = 25, window_s: Optional[float] = None):
        self.window = window
        self.window_s = window_s if window_s is not None else window * DESIGN_DT_S
        self._buf: list = []          # (timestamp, value)
        self._synthetic_t = 0.0

    def reset(self):
        self._buf.clear()
        self._synthetic_t = 0.0

    @property
    def samples(self) -> list:
        return [v for _, v in self._buf]

    def push(self, value: Optional[float], now: Optional[float] = None) -> Optional[float]:
        if value is None:
            return self.score()
        if now is None:
            self._synthetic_t += DESIGN_DT_S
            now = self._synthetic_t
        self._buf.append((now, value))
        cutoff = now - self.window_s
        while len(self._buf) > self.MIN_KEEP and self._buf[0][0] < cutoff:
            self._buf.pop(0)
        if len(self._buf) > 400:
            del self._buf[:-400]
        return self.score()

    def score(self) -> Optional[float]:
        vals = self.samples
        n = len(vals)
        if n < self.MIN_KEEP:
            return None
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return math.sqrt(var)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))
