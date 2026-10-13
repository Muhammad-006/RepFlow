"""
session.py — one athlete's encrypted session.

Holds, for the life of one set and no longer:
  * the keys negotiated with that phone (in bytearrays, so they can be
    zeroed on the way out),
  * the exercise coach and its rep history,
  * the orientation lock,
  * a replay window over frame counters,
  * a running privacy ledger (what arrived, what was kept — the answer is
    always "nothing").

What it deliberately does not hold: any frame, any decoded image, any
annotated copy, any file handle, the device model, or the phone's IP.
There is no code path in this class that writes to disk.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import crypto_box as cb
import exercises
from feedback import FeedbackEngine
from pose_engine import KEYPOINT_NAMES, PoseResult, Track

ROTATIONS = (0, 90, 180, 270)


# ----------------------------------------------------------------------
def rot_point(r: int, x: float, y: float) -> Tuple[float, float]:
    if r == 90:
        return 1.0 - y, x
    if r == 180:
        return 1.0 - x, 1.0 - y
    if r == 270:
        return y, 1.0 - x
    return x, y


def unrot_point(r: int, u: float, v: float) -> Tuple[float, float]:
    return rot_point((360 - r) % 360, u, v)


class AutoOrient:
    """Works out which way is up in the camera stream.

    The phone's accelerometer is the primary signal and is right almost
    always; vision probing is the fallback for a phone lying flat on a
    bench, where gravity tells you nothing useful.

    Probing costs extra inference passes, so it only ever runs while the
    locked rotation is finding nothing — that is, while the app is
    already useless — and it stops the instant a rotation starts working.

    It sweeps all four rotations on one frame rather than trying one
    alternative every few frames. Dribbling was the tempting design and
    it is the wrong one: at one alternative per six frames, cycling three
    alternatives, needing three confirmations, a phone lying flat on a
    bench took over six seconds to find the athlete. Sweeping costs three
    extra forward passes on a handful of frames — about 25 ms each, on
    frames that were already producing nothing — and finds the right
    rotation in under half a second.
    """

    # Consecutive misses on the locked rotation before we stop believing
    # the phone's gravity reading. About a second at 10 fps.
    PROBE_AFTER_MISSES = 8
    # Before the very first successful detection of a set, be much more
    # impatient. A phone that has never once found the athlete has no
    # evidence its rotation is right, and the athlete is standing there
    # waiting — this is the moment where a second of nothing reads as
    # "the app is broken". After the first hit, the ordinary patience
    # above applies, because by then a miss is more likely to be someone
    # stepping out of shot than the picture being the wrong way up.
    COLD_START_MISSES = 2
    # A miss *streak* is not the only symptom of a wrong rotation, and on
    # its own it is a trap: a person rotated 90 degrees is still a person
    # as far as the model is concerned, so a sideways frame detects
    # sporadically — roughly one frame in four. Every one of those resets
    # a streak counter, so a streak-only trigger never fires and the app
    # sits at 25% tracking forever, which is exactly what it did. So we
    # also watch the hit *rate*: a rotation that works finds the athlete
    # almost every frame, and anything much worse is worth questioning.
    # The window is short so a wrong rotation is caught in about a second,
    # and the bar is set low enough (half the frames missed, not a third)
    # that an athlete who is briefly occluded — a hand across the lens, a
    # dropped frame, a moment side-on — does not start a hunt for a
    # rotation problem that isn't there.
    WINDOW = 8
    POOR_HIT_RATE = 0.6
    # Don't sweep on every frame while lost: leave room for the locked
    # rotation to recover on its own (the athlete may just have stepped
    # out of shot, which no amount of rotating will fix).
    SWEEP_EVERY_N = 2
    # When a sweep finds nobody in *any* rotation this many times in a
    # row, the room is empty — rotating the frame cannot fix that, and
    # paying three extra forward passes every other frame to keep proving
    # it is how a laptop fan ends up screaming at an empty garage. Back
    # off to an occasional look until someone appears.
    EMPTY_SWEEPS_BEFORE_BACKOFF = 8
    SLOW_SWEEP_EVERY_N = 15
    SWITCH_AFTER_HITS = 2
    # How much better an alternative has to look before we believe it.
    SWITCH_MARGIN = 1.15
    # Gravity is right almost always. If vision has had to overrule it
    # this many times in one set, the hint is simply unusable on this
    # setup — keep the rotation vision found and stop paying for sweeps.
    MAX_VISION_SWITCHES = 2

    def __init__(self, force: Optional[int] = None):
        self.force = force
        self.locked: Optional[int] = force if force is not None else 0
        self.source = "manual" if force is not None else "gravity"
        self.misses = 0
        self.sweeps = 0
        self._last_hint: Optional[int] = None
        self._frames = 0
        self._switches = 0
        self._recent: List[bool] = []                    # locked rotation only
        self._alt_hits: Dict[int, int] = {r: 0 for r in ROTATIONS}
        self._alt_score: Dict[float, float] = {r: 0.0 for r in ROTATIONS}
        self._ever_hit = False                           # any detection at all yet?
        self._sweep_frame: Optional[int] = None          # frame the last sweep ran on
        self._sweep_hit = False                          # did it find anyone, anywhere?
        self._empty_sweeps = 0
        self._was_searching = False

    def _new_episode(self, rot: int, source: str) -> None:
        self.locked, self.source, self.misses = rot, source, 0
        self._recent = []
        self._alt_hits = {r: 0 for r in ROTATIONS}
        self._alt_score = {r: 0.0 for r in ROTATIONS}
        self._empty_sweeps = 0

    @property
    def hit_rate(self) -> float:
        if not self._recent:
            return 1.0
        return sum(self._recent) / float(len(self._recent))

    @property
    def searching(self) -> bool:
        """Is the locked rotation doing badly enough to be worth testing?"""
        if not self._ever_hit and self.misses >= self.COLD_START_MISSES:
            return True
        if self.misses >= self.PROBE_AFTER_MISSES:
            return True
        return len(self._recent) >= self.WINDOW and self.hit_rate < self.POOR_HIT_RATE

    def candidates(self, hint: int) -> List[int]:
        self._frames += 1
        if self.force is not None:
            self.locked, self.source = self.force, "manual"
            return [self.force]
        if hint != self._last_hint and self.source != "vision":
            # A new gravity reading re-aims the lock — unless vision has
            # already had to overrule gravity for this phone, in which
            # case the hint has proved untrustworthy and a fresh one is
            # no more believable than the last.
            self._last_hint = hint
            self._new_episode(hint, "gravity")

        # Score the previous sweep before deciding on another one. A sweep
        # that found nobody in any of the four rotations says the room is
        # empty, not that the phone is the wrong way up.
        if self._sweep_frame is not None and self._sweep_frame != self._frames:
            self._empty_sweeps = 0 if self._sweep_hit else self._empty_sweeps + 1
            self._sweep_frame, self._sweep_hit = None, False

        every = (self.SWEEP_EVERY_N
                 if self._empty_sweeps < self.EMPTY_SWEEPS_BEFORE_BACKOFF
                 else self.SLOW_SWEEP_EVERY_N)
        searching = self.searching
        # The frame on which things first start looking wrong is the most
        # valuable one to sweep — waiting for the counter to line up with
        # the modulus threw away up to `every` frames for nothing.
        due = searching and (not self._was_searching or self._frames % every == 0)
        self._was_searching = searching
        if (due and self._switches < self.MAX_VISION_SWITCHES):
            self.sweeps += 1
            self._sweep_frame, self._sweep_hit = self._frames, False
            return [self.locked] + [r for r in ROTATIONS if r != self.locked]
        return [self.locked]

    def observe(self, rot: int, detected: bool, quality: float = 0.0):
        """Record how one rotation did on one frame.

        `quality` matters as much as `detected`: the wrong rotation tends
        to find *a* person with poor, scattered keypoints, so comparing
        confidence separates "found the athlete" from "found something
        person-shaped lying on its side".
        """
        if self.force is not None:
            return
        score = float(quality) if detected else 0.0
        if detected:
            self._ever_hit = True
            if self._sweep_frame == self._frames:
                self._sweep_hit = True
        if rot == self.locked:
            self._recent.append(bool(detected))
            if len(self._recent) > self.WINDOW:
                self._recent.pop(0)
            self._alt_score[rot] = 0.7 * self._alt_score[rot] + 0.3 * score
            if detected:
                self.misses = 0
            else:
                self.misses += 1
            return

        self._alt_score[rot] = 0.7 * self._alt_score[rot] + 0.3 * score
        if not detected:
            return
        self._alt_hits[rot] += 1
        better = self._alt_score[rot] > self._alt_score[self.locked] * self.SWITCH_MARGIN
        if self._alt_hits[rot] >= self.SWITCH_AFTER_HITS and better:
            self._switches += 1
            self._new_episode(rot, "vision")


# ----------------------------------------------------------------------
class ClockSync:
    """Turns the phone's timestamps into ages this server can trust.

    The phone stamps each frame with its own `time.time()`. That clock is
    not this machine's clock: two devices that both think they are right
    are routinely seconds apart, and the phone can be either ahead or
    behind. Subtracting one from the other therefore does not give the
    age of the frame — it gives the age plus an unknown constant offset.

    v3.0 subtracted them directly and dropped anything over 600 ms. A
    phone whose clock sat one second behind the laptop's had *every*
    frame classified as stale, so the model never ran, and the app showed
    a camera preview with nothing tracking on it and no error anywhere.
    That is the bug this class exists to make impossible.

    The fix is the standard one-way-delay trick: the offset is unknown but
    nearly constant, and the fastest frame in any recent window is the one
    that spent the least time in the air. So `offset = min(t_recv - t_sent)`
    over a rolling window is a good estimate of "clock offset plus the
    network's floor", and `age = (t_recv - t_sent) - offset` is how much
    later than the *best* frame this one arrived. That is what staleness
    actually means here, and it is immune to any constant skew.

    The window slides so a phone that re-syncs mid-set (or drifts) is
    tracked rather than fought, and until it has enough samples nothing is
    ever dropped — a set must never open by discarding frames.
    """

    WARMUP = 12          # frames before staleness is enforced at all
    WINDOW_S = 20.0      # re-estimate the floor over this much history
    MIN_SAMPLES = 6
    # A frame later than this than the phone's own baseline is too old to
    # coach from. This is the one definition of "stale" in the codebase;
    # server.py imports it rather than keeping a second copy.
    LATE_S = 0.60
    # Consecutive late verdicts before we stop believing our own estimate.
    # Late frames come in ones and twos when a link hiccups; an unbroken
    # run of them means the phone's clock stepped (an NTP correction, a
    # timezone change, the user landing) and the baseline now describes a
    # clock that no longer exists. Re-seeding beats spending twenty
    # seconds waiting for the old samples to age out of the window while
    # the athlete watches a frozen screen.
    RESYNC_AFTER = 5

    # How far apart frames have to arrive before we conclude the phone is
    # round-trip-bound — i.e. it is not holding a queue of frames, it is
    # waiting on our reply before it can capture the next one. See
    # `round_trip_bound` below; this is what makes dropping a late frame
    # the wrong move rather than the right one.
    ROUND_TRIP_BOUND_INTERVAL_S = 0.35

    def __init__(self):
        self._samples: List[Tuple[float, float]] = []   # (t_recv, raw_delta)
        self.offset = 0.0
        self.n = 0
        self.ready = False
        self.resyncs = 0
        self._late_run = 0
        self._last_recv: Optional[float] = None
        self.interval_s = 0.0      # smoothed gap between arrivals

    @property
    def round_trip_bound(self) -> bool:
        """True when frames arrive far enough apart that the phone is
        clearly waiting on us rather than queueing.

        This matters because dropping a late frame only pays off if a
        FRESHER frame is already on its way. The phone holds exactly one
        frame in flight — it cannot capture the next one until this reply
        lands — so once the round trip is the thing setting the frame
        rate, refusing to coach on a late frame buys nothing and costs
        everything: the athlete pays the full round trip and gets an
        empty answer, and with a run of drops the skeleton simply
        vanishes for several seconds. Coaching on a slightly old frame is
        strictly better, and the cue layer already refuses to SPEAK from
        stale data (feedback.MAX_DECISION_AGE_S), so the risk that
        dropping was protecting against is already handled one layer
        down.
        """
        return self.interval_s > self.ROUND_TRIP_BOUND_INTERVAL_S

    def observe(self, t_recv: float, t_sent: float) -> float:
        """Record one frame and return its age in seconds (>= 0)."""
        raw = t_recv - t_sent
        if self._last_recv is not None:
            gap = t_recv - self._last_recv
            if 0.0 < gap < 10.0:
                self.interval_s = (gap if not self.interval_s
                                   else 0.8 * self.interval_s + 0.2 * gap)
        self._last_recv = t_recv
        self._samples.append((t_recv, raw))
        self.n += 1
        cutoff = t_recv - self.WINDOW_S
        if len(self._samples) > 64 or (self._samples and self._samples[0][0] < cutoff):
            self._samples = [s for s in self._samples if s[0] >= cutoff][-64:]
        floor = min(s[1] for s in self._samples)
        # Ratchet downwards quickly (a faster frame is proof of a lower
        # floor) and upwards slowly (so one lucky frame early in the set
        # does not make everything after it look stale forever).
        if floor < self.offset or not self.ready:
            self.offset = floor
        else:
            self.offset += 0.05 * (floor - self.offset)
        self.ready = self.n >= self.WARMUP and len(self._samples) >= self.MIN_SAMPLES

        age = max(0.0, raw - self.offset)
        if age > self.LATE_S:
            self._late_run += 1
            if self._late_run >= self.RESYNC_AFTER:
                self._resync(t_recv, raw)
                return 0.0
        else:
            self._late_run = 0
        return age

    def _resync(self, t_recv: float, raw: float) -> None:
        """Throw the history away and start again from this frame.

        Warm-up restarts too, so nothing is dropped while the new
        baseline is being established — the failure this whole class
        exists to prevent is dropping frames on a guess.
        """
        self._samples = [(t_recv, raw)]
        self.offset = raw
        self.n = 1
        self.ready = False
        self._late_run = 0
        self.resyncs += 1

    @property
    def skew_s(self) -> float:
        """The phone's clock minus this machine's, in seconds.

        Negative means the phone believes it is earlier than we do. The
        network's one-way floor is folded in and is milliseconds against
        skews that are typically whole seconds, so this is reported with
        the sign a reader expects rather than as a raw internal offset.
        """
        return -self.offset


class ReplayWindow:
    """Rejects a frame counter we have already accepted.

    Frames can legitimately arrive slightly out of order, so this keeps a
    sliding window rather than demanding a strictly increasing counter.
    """

    SIZE = 256

    def __init__(self):
        self.highest = -1
        self._seen: set = set()

    def check_and_mark(self, counter: int) -> bool:
        if counter <= self.highest - self.SIZE:
            return False
        if counter in self._seen:
            return False
        self._seen.add(counter)
        self.highest = max(self.highest, counter)
        if len(self._seen) > self.SIZE * 2:
            cutoff = self.highest - self.SIZE
            self._seen = {c for c in self._seen if c > cutoff}
        return True


# ----------------------------------------------------------------------
class PrivacyLedger:
    """What the server can honestly say it did with your video.

    Every number in here is incremented at the exact point the thing it
    counts happens, so it is a record rather than a claim. `frames_stored`
    and `bytes_written_to_disk` have no code anywhere that can increment
    them; they are in the ledger so the app can display a zero that means
    something.
    """

    def __init__(self):
        self.frames_received = 0
        self.frames_decrypted = 0
        self.frames_rejected = 0
        self.bytes_received = 0
        self.frames_stored = 0              # structurally always 0
        self.bytes_written_to_disk = 0      # structurally always 0
        self.plaintext_lifetime_ms = 0.0
        self.created = time.time()
        self.keys_destroyed_at: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "frames_decrypted": self.frames_decrypted,
            "frames_rejected": self.frames_rejected,
            "mb_received": round(self.bytes_received / 1e6, 2),
            "frames_stored": self.frames_stored,
            "bytes_written_to_disk": self.bytes_written_to_disk,
            "avg_plaintext_lifetime_ms": round(
                self.plaintext_lifetime_ms / self.frames_decrypted, 1)
            if self.frames_decrypted else 0.0,
            "keys_destroyed": self.keys_destroyed_at is not None,
        }


# ----------------------------------------------------------------------
class Session:
    IDLE_TTL_S = 10 * 60

    def __init__(self, session_id: str, keys: dict, exercise: str = exercises.DEFAULT):
        self.id = session_id
        self.short_id = session_id[:6].upper()
        self.lock = threading.Lock()
        self.created = time.time()
        self.last_seen = time.time()

        # Keys live in bytearrays so `wipe()` can actually overwrite them.
        self.keys = {k: bytearray(v) for k, v in keys.items()}
        self.replay = ReplayWindow()
        self.privacy = PrivacyLedger()
        self.clock = ClockSync()
        self.stale_drops = 0
        self.last_drop_reason: Optional[str] = None
        self.detect_reason: Optional[str] = None

        self.exercise_key = exercise
        self.coach = exercises.make(exercise)
        self.feedback = FeedbackEngine()
        self.orient = AutoOrient()
        self.target = 0
        self.goal_announced = False

        self.frames = 0
        self.detected_frames = 0
        self.lost_since: Optional[float] = None
        self.fps = 0.0
        self.infer_ms = 0.0
        self.prev_box = None
        self.prev_rot: Optional[int] = None
        # The engine is shared between phones; the ROI lock is not, so it
        # lives here and is handed to the engine one frame at a time.
        self.track = Track()
        self._last_frame_t: Optional[float] = None
        self.set_start = time.time()
        self.ended = False
        self.last: dict = {}

    # ------------------------------------------------------------------
    def key(self, name: str) -> bytes:
        return bytes(self.keys[name])

    def reset(self, exercise: Optional[str] = None, target: int = 0):
        now = time.time()
        if exercise and exercise != self.exercise_key:
            self.exercise_key = exercise
            self.coach = exercises.make(exercise)
        self.coach.reset(now)
        if hasattr(self.coach, "set_target"):
            self.coach.set_target(target)
        self.feedback.reset()
        self.orient = AutoOrient()
        self.target = max(0, int(target or 0))
        self.goal_announced = False
        self.frames = 0
        self.detected_frames = 0
        self.lost_since = None
        self.fps = 0.0
        self.infer_ms = 0.0
        self.prev_box = None
        self.prev_rot = None
        self.track.clear()
        self._last_frame_t = None
        self.set_start = now
        self.ended = False
        self.last = {}
        self.stale_drops = 0
        self.last_drop_reason = None
        # The clock estimate deliberately survives a reset: it describes
        # this phone and this laptop, not this set, and throwing it away
        # would re-open the warm-up window at the start of every set.

    def destroy(self):
        """Zero every key. After this the session can decrypt nothing —
        including anything an attacker recorded earlier."""
        for buf in self.keys.values():
            cb.wipe(buf)
        self.keys.clear()
        self.privacy.keys_destroyed_at = time.time()
        self.prev_box = None
        self.track.clear()

    @property
    def expired(self) -> bool:
        return time.time() - self.last_seen > self.IDLE_TTL_S

    # ------------------------------------------------------------------
    def update(self, pose: Optional[PoseResult], rot: int, display_rot: int,
               frame_aspect: float, capture_age_s: float, server_ms: float) -> dict:
        now = time.time()
        self.last_seen = now
        self.frames += 1
        if self._last_frame_t is not None:
            dt = max(1e-3, now - self._last_frame_t)
            inst = 1.0 / dt
            self.fps = 0.8 * self.fps + 0.2 * inst if self.fps else inst
        self._last_frame_t = now
        if pose is not None:
            self.infer_ms = (0.8 * self.infer_ms + 0.2 * pose.inference_ms
                             if self.infer_ms else pose.inference_ms)

        detected = bool(pose and pose.is_detected)

        # Why is nothing tracking? v3.0 could not answer this, which is
        # how a session where every frame was being discarded looked
        # exactly like a session where the athlete was out of shot. The
        # reason travels back to the phone on every frame.
        reason = None
        if not detected:
            if pose is None:
                reason = "no_pose"
            elif pose.people_in_frame == 0:
                reason = "nobody_in_frame"
            elif pose.quality < 0.15:
                reason = "too_far_or_dark"
            else:
                reason = "partial_body"
        self.detect_reason = reason

        kp = None
        if detected:
            self.detected_frames += 1
            self.prev_box = pose.box
            # Un-squash normalized coordinates so angles are measured in
            # real proportions rather than in a stretched unit square.
            kp = {k: (v[0] * frame_aspect, v[1], v[2]) for k, v in pose.keypoints.items()}
            self.lost_since = None
        elif self.lost_since is None:
            self.lost_since = now

        tick = self.coach.update(kp, now, detected)

        # --- assemble the ranked cue list --------------------------------
        candidates: List[str] = list(tick.cues)
        if not detected:
            lost_for = now - (self.lost_since or now)
            if lost_for > 2.5:
                candidates.insert(0, "lost_tracking")
        if (self.target and tick.count >= self.target and not self.goal_announced
                and self.coach.UNIT == "reps"):
            self.goal_announced = True
            candidates.insert(0, "goal_reached")

        # A completed rep always gets its word in — that is the heartbeat
        # of the whole experience — but nothing else jumps the queue.
        force = tick.record is not None and bool(candidates)
        cue_id = self.feedback.select(candidates, decision_age_s=capture_age_s,
                                      force=force, now=now)

        # --- keypoints, rotated back into the phone's own frame ----------
        kp_display = None
        if detected and pose is not None:
            kp_display = []
            for name in KEYPOINT_NAMES:
                x, y, c = pose.keypoints[name]
                rx, ry = unrot_point(rot, x, y)
                dx, dy = rot_point(display_rot, rx, ry)
                kp_display.append([round(dx, 4), round(dy, 4), round(c, 2)])

        result = {
            "exercise": self.exercise_key,
            "unit": self.coach.UNIT,
            "count": tick.count,
            "clean": sum(1 for r in self.coach.records if r.clean),
            "stage": tick.stage,
            "is_detected": detected,
            "is_calibrated": tick.is_calibrated,
            "calibration_pct": round(tick.calibration_pct, 2),
            "primary": round(tick.primary, 1) if tick.primary is not None else None,
            "primary_label": self.coach.PRIMARY_LABEL,
            "secondary": round(tick.secondary, 1) if tick.secondary is not None else None,
            "secondary_label": self.coach.SECONDARY_LABEL,
            "faults": list(tick.faults),
            "quality": tick.quality,
            "hold_s": tick.hold_s,
            "rep_completed": tick.record is not None,
            "last_rep": _rep_dict(tick.record) if tick.record else None,
            "form_score": _avg_score(self.coach.records),
            "cue_id": cue_id,
            "keypoints": kp_display,
            "confidence": round(pose.quality, 2) if pose else 0.0,
            "rotation": rot,
            "fps": round(self.fps, 1),
            "inference_ms": round(pose.inference_ms, 1) if pose else None,
            "server_ms": round(server_ms, 1),
            "capture_age_ms": round(capture_age_s * 1000.0),
            "target": self.target,
            # --- diagnostics: why tracking is or isn't working ---------
            "detect_reason": reason,
            "people_in_frame": pose.people_in_frame if pose else 0,
            "track_mode": pose.mode if pose else None,
            "tracking_pct": round(100 * self.detected_frames / self.frames) if self.frames else 0,
            "clock_skew_ms": round(self.clock.skew_s * 1000.0),
        }
        result.update({k: round(v, 1) for k, v in tick.extra.items()})
        self.last = result
        return result

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        now = time.time()
        if hasattr(self.coach, "finalize"):
            self.coach.finalize(now)
        s = self.coach.summary(now - self.set_start)
        s.update({
            "exercise": self.exercise_key,
            "exercise_name": self.coach.NAME,
            "frames": self.frames,
            "tracking_pct": round(100 * self.detected_frames / self.frames) if self.frames else 0,
            "target": self.target,
            "cues": self.feedback.stats,
            "privacy": self.privacy.as_dict(),
        })
        return s


def _rep_dict(r) -> dict:
    d = {"rep": r.index, "score": r.score, "clean": r.clean, "fault": r.fault,
         "duration": round(r.duration, 2)}
    d.update({k: round(v, 1) for k, v in r.metrics.items()})
    return d


def _avg_score(records) -> Optional[int]:
    if not records:
        return None
    return int(round(sum(r.score for r in records) / len(records)))
