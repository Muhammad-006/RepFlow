"""
pose_engine.py — RepFlow's pose-estimation engine (17 COCO keypoints).

The design goal is a pose estimate in under 8 ms on an ordinary laptop
CPU, with no GPU and no cloud call. Three things get us there, and the
third is the one that matters:

1.  A small input resolution.
    Inference cost on a fully-convolutional detector scales with input
    *area*, so 640 -> 320 is a 4x saving and 640 -> 256 is 6.25x. The
    repo ships the same yolo11n-pose weights exported at 192, 256, 320
    and 640 so the size is a runtime choice rather than a rebuild.

2.  Picking that resolution by measuring, not guessing.
    `imgsz="auto"` benchmarks the shipped models on the machine it is
    actually running on and keeps the *largest* one that fits the latency
    budget — best accuracy that still makes the deadline. A fast laptop
    gets 320; a netbook drops to 192 on its own rather than falling apart.

3.  Running the model on the athlete, not on the room.
    A phone propped across a garage sees a person occupying maybe 15% of
    the frame. Letterboxing that to 256x256 spends 85% of the model's
    pixels on the floor and the wall. So once the athlete is located, the
    next frame is cropped to a padded square around them and *that* is
    what the model sees. The body then fills the input, and a 256 crop
    resolves it about as well as a 640 full frame:

        full frame  @640   conf 0.904    ~55 ms  (2-core reference box)
        full frame  @256   conf 0.841    ~10 ms
        ROI crop    @256   conf 0.903    ~10 ms   <- same accuracy, 5.5x faster

    The crop is re-verified against the full frame periodically and
    whenever it comes up empty, so a lock can never silently drift onto
    an empty patch of wall.

Two interchangeable backends are supported:

* ONNX Runtime (default, recommended for laptops). Runs the bundled
  models, so the server works fully offline. Needs only `numpy`,
  `opencv-python` and `onnxruntime`.
* Ultralytics / PyTorch (optional, for NVIDIA GPUs). Used with
  `--backend ultralytics`, or when `auto` finds a CUDA GPU.

Both return the same PoseResult, so everything downstream
(exercises/*.py, session.py) doesn't care which one ran.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("repflow.pose")

KEYPOINT_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

CORE_JOINTS: List[str] = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges (pairs of keypoint indices) -- used for drawing overlays.
SKELETON: List[Tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head -> shoulders
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Default latency budget for one forward pass, in milliseconds.
DEFAULT_BUDGET_MS = 8.0

_ZERO_KEYPOINTS: Dict[str, Tuple[float, float, float]] = {
    name: (0.0, 0.0, 0.0) for name in KEYPOINT_NAMES
}


@dataclass
class PoseResult:
    keypoints: Dict[str, Tuple[float, float, float]]  # name -> (x_norm, y_norm, conf)
    is_detected: bool
    high_conf_count: int
    dominant_side: str                                # 'left' | 'right' | 'unknown'
    quality: float = 0.0                              # mean confidence of core joints (0..1)
    box: Optional[Tuple[float, float, float, float]] = None  # normalized x1,y1,x2,y2
    inference_ms: float = 0.0
    people_in_frame: int = 0
    mode: str = "full"                                # 'full' | 'roi' | 'roi->full'


@dataclass
class EngineInfo:
    backend: str
    model: str
    device: str
    input_size: int
    keypoints: int = 17
    warmup_ms: float = 0.0
    bench_ms: float = 0.0          # measured median forward pass at startup
    budget_ms: float = DEFAULT_BUDGET_MS
    finder_size: int = 0           # input size of the whole-frame pass
    finder_ms: float = 0.0         # what that pass costs, measured here
    extra: dict = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return bool(self.bench_ms) and self.bench_ms <= self.budget_ms


@dataclass
class Track:
    """Per-athlete ROI state. One of these lives on each Session.

    The engine is shared between phones; this is the little bit of state
    that is emphatically not, so it is passed in rather than stored.
    """
    box: Optional[Tuple[float, float, float, float]] = None   # normalized
    misses: int = 0
    frames_since_full: int = 0
    hits: int = 0

    def clear(self) -> None:
        self.box = None
        self.misses = 0
        self.frames_since_full = 0
        self.hits = 0


def _pretty_model_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"-\d+$", "", base)
    return base.replace("yolo11", "YOLO11").replace("yolov8", "YOLOv8").replace("-pose", "-Pose")


def _shipped_models() -> List[Tuple[int, str]]:
    """Every ONNX pose model in models/, as (input_size, path), small first.

    The size is read from the filename when it carries one, so dropping a
    `yolo11n-pose-288.onnx` into models/ is all it takes to offer a new
    option to the auto-picker.
    """
    found: List[Tuple[int, str]] = []
    for path in glob.glob(os.path.join(MODELS_DIR, "*.onnx")):
        m = re.search(r"-(\d{2,4})\.onnx$", path)
        found.append((int(m.group(1)) if m else 640, path))
    return sorted(found)


class PoseEngine:
    """Loads a pose model once and runs it on BGR frames."""

    # How often the ROI lock is re-checked against the whole frame. At
    # 10 fps this is a full-frame sweep roughly twice a second, which is
    # fast enough to pick up the athlete moving across the frame and
    # cheap enough not to show up in the average.
    FULL_RESCAN_EVERY = 18
    # Give up on a lock after this many ROI misses in a row.
    MAX_ROI_MISSES = 2
    # How much bigger than the last box the crop is. A push-up's arms
    # swing well outside the previous frame's box, so this is generous.
    ROI_PAD = 1.6

    # The smallest input a *full-frame* pass may use. Finding a person and
    # measuring their joints are not the same problem and they do not have
    # the same cheapest answer, which is the mistake that made this app
    # see nothing. Measured on a 1280x720 clip of one athlete walking from
    # about three metres in to arm's length (tools/verify_tracking.py):
    #
    #   full frame @192   24/40 frames   blind below ~44% of frame height
    #   full frame @256   40/40 frames   found the athlete at 30% and up
    #   full frame @320   40/40 frames
    #
    # 192 was fast and useless: an athlete who sets their phone against a
    # wall and steps back — the normal way to film a push-up — is a person
    # roughly a third of the frame tall, which is exactly where it goes
    # blind. On the ROI crop, though, the athlete fills the input and the
    # small model is as good as the big one:
    #
    #   ROI crop @192     median joint error 5.0 px   (vs a 640px reference)
    #   ROI crop @256     median joint error 4.8 px
    #   ROI crop @320     median joint error 4.2 px
    #
    # So: a finder of at least this size on the whole frame, and whatever
    # fits the latency budget on the crop. Steady-state tracking pays the
    # small number; acquisition pays the larger one, on the frames that
    # would otherwise have produced nothing at all.
    MIN_FINDER_SIZE = 256
    # A box at least this tall (as a fraction of frame height) is an
    # athlete the tracker can find on the whole frame by itself — the
    # measurements above put the small model's blind spot below ~44%.
    TRACKER_OK_BOX_FRAC = 0.5

    def __init__(
        self,
        backend: str = "auto",
        model: Optional[str] = None,
        min_keypoint_confidence: float = 0.30,
        min_detected_joints: int = 5,
        det_confidence: float = 0.30,
        threads: Optional[int] = None,
        imgsz: str | int = "auto",
        budget_ms: float = DEFAULT_BUDGET_MS,
        roi: bool = True,
    ):
        self.min_confidence = min_keypoint_confidence
        self.min_detected_joints = min_detected_joints
        self.det_confidence = det_confidence
        self.budget_ms = float(budget_ms)
        self.roi_enabled = bool(roi)
        self._impl = None          # tracker: runs on the athlete crop
        self._finder = None        # runs on the whole frame (may be the same)
        self.info: Optional[EngineInfo] = None
        self._bench_ms: Optional[float] = None
        self._bench_table: Dict[int, float] = {}
        self._threads = threads

        backend = (backend or "auto").lower()
        errors: List[str] = []
        order = {
            "auto": ["ultralytics-gpu", "onnx", "ultralytics"],
            "onnx": ["onnx"],
            "ultralytics": ["ultralytics"],
        }.get(backend, ["onnx"])

        for choice in order:
            try:
                if choice == "onnx":
                    self._impl = _OnnxBackend(
                        self._pick_onnx_model(model, imgsz), threads)
                elif choice == "ultralytics-gpu":
                    self._impl = _UltralyticsBackend(
                        model if (model and model.endswith(".pt")) else "yolo11m-pose.pt",
                        require_gpu=True)
                elif choice == "ultralytics":
                    self._impl = _UltralyticsBackend(
                        model if (model and model.endswith(".pt")) else "yolo11n-pose.pt")
                break
            except Exception as e:  # try the next backend
                errors.append(f"{choice}: {e}")
                self._impl = None

        if self._impl is None:
            raise RuntimeError("No pose backend could be loaded:\n  " + "\n  ".join(errors))

        self.info = self._impl.info
        self.info.budget_ms = self.budget_ms
        self.info.extra.setdefault("roi_crop", self.roi_enabled)
        if self._bench_ms is not None:
            self.info.bench_ms = self._bench_ms
            self.info.extra["benchmark_ms"] = self._bench_table

        self._finder = self._load_finder()
        self.info.finder_size = getattr(self._finder, "size", self.info.input_size)

        # Warm-up so the first real frame from a phone isn't the slow one,
        # then a short measured run so `info.bench_ms` is a real number
        # this machine produced rather than a claim.
        probe = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        try:
            for _ in range(3):
                self._impl.infer(probe, self.det_confidence)
            if self._finder is not self._impl:
                for _ in range(2):
                    self._finder.infer(probe, self.det_confidence)
        except Exception as e:
            log.warning("Warm-up inference failed: %s", e)
        self.info.warmup_ms = (time.perf_counter() - t0) * 1000.0
        if not self.info.bench_ms:
            self.info.bench_ms = _time_model(self._impl, probe, self.det_confidence, runs=8)
        self.info.finder_ms = (self.info.bench_ms if self._finder is self._impl
                               else _time_model(self._finder, probe, self.det_confidence, runs=6))

    # ------------------------------------------------------------------
    def _load_finder(self):
        """The model used for whole-frame passes.

        Usually this is the tracker itself — there is no point holding two
        copies of the same weights. It differs only when the budget drove
        the tracker below `MIN_FINDER_SIZE`, where a full-frame pass stops
        being able to find a distant athlete at all (see the note on that
        constant). Accuracy wins that trade: a slower frame is a cost, a
        frame that finds nobody is the app not working.
        """
        if not isinstance(self._impl, _OnnxBackend):
            return self._impl                      # ultralytics: one model, one cost
        if self._impl.size >= self.MIN_FINDER_SIZE:
            return self._impl

        bigger = [(s, p) for s, p in _shipped_models() if s >= self.MIN_FINDER_SIZE]
        if not bigger:
            log.warning("No model of at least %dpx is shipped; a distant athlete "
                        "may not be found at all.", self.MIN_FINDER_SIZE)
            return self._impl
        size, path = bigger[0]
        try:
            finder = _OnnxBackend(path, self._threads)
        except Exception as e:
            log.warning("Could not load the %dpx finder (%s); staying on %dpx.",
                        size, e, self._impl.size)
            return self._impl
        log.info("finder %dpx (whole frame) · tracker %dpx (athlete crop)",
                 size, self._impl.size)
        return finder

    # ------------------------------------------------------------------
    def _pick_onnx_model(self, model: Optional[str], imgsz: str | int) -> str:
        """Choose which shipped ONNX to load.

        An explicit path or size always wins. Otherwise every shipped
        model is timed on this machine and the largest one inside the
        latency budget is kept, because within the budget bigger input =
        better keypoints. If none fits, the fastest is used and the
        shortfall is logged rather than hidden.
        """
        if model and model.endswith(".onnx"):
            return model

        shipped = _shipped_models()
        if not shipped:
            raise FileNotFoundError(f"no ONNX pose models found in {MODELS_DIR}")

        if isinstance(imgsz, int) or (isinstance(imgsz, str) and imgsz.isdigit()):
            want = int(imgsz)
            exact = [p for s, p in shipped if s == want]
            if exact:
                return exact[0]
            nearest = min(shipped, key=lambda sp: abs(sp[0] - want))
            log.warning("No %dpx model shipped; using %dpx instead.", want, nearest[0])
            return nearest[1]

        # --- auto: measure, then choose ---------------------------------
        probe = np.zeros((480, 640, 3), dtype=np.uint8)
        timings: List[Tuple[int, str, float]] = []
        for size, path in shipped:
            try:
                impl = _OnnxBackend(path, None)
                ms = _time_model(impl, probe, self.det_confidence, runs=6)
                timings.append((size, path, ms))
                log.info("benchmark %3dpx -> %5.1f ms", size, ms)
            except Exception as e:
                log.warning("could not benchmark %s: %s", os.path.basename(path), e)
            finally:
                impl = None

        if not timings:
            return shipped[0][1]

        fits = [t for t in timings if t[2] <= self.budget_ms]
        chosen = max(fits, key=lambda t: t[0]) if fits else min(timings, key=lambda t: t[2])
        if not fits:
            log.warning(
                "No shipped model meets the %.0f ms budget on this CPU; using %dpx at "
                "%.1f ms. Raise --latency-budget, or pass --threads to use more cores.",
                self.budget_ms, chosen[0], chosen[2])
        self._bench_ms = chosen[2]
        self._bench_table = {s: round(ms, 2) for s, _, ms in timings}
        return chosen[1]

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, prev_box=None,
                      track: Optional[Track] = None,
                      probe: bool = False) -> Optional[PoseResult]:
        """Run pose on one BGR frame.

        `track` carries the ROI lock for one athlete and is updated in
        place. `prev_box` is the older single-value form of the same idea
        and is still accepted so nothing downstream had to change at once.

        `probe=True` means the caller only wants to know *whether* a
        person is there — it is testing a camera rotation, not coaching
        from the answer — so the keypoint-refinement pass is skipped and
        the call costs one forward pass instead of two.
        """
        if frame_bgr is None or self._impl is None:
            return None
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None

        if track is None:
            track = Track(box=prev_box)
        elif prev_box is not None and track.box is None:
            track.box = prev_box

        use_roi = (
            self.roi_enabled
            and track.box is not None
            and track.hits > 0
            and track.frames_since_full < self.FULL_RESCAN_EVERY
        )

        t0 = time.perf_counter()
        mode = "full"
        people: List[tuple] = []
        offset = (0, 0)
        scale_src = (w, h)

        if use_roi:
            crop, ox, oy = self._crop(frame_bgr, track.box)
            if crop is None or not crop.size:
                # `_crop` declines when the athlete already fills the
                # frame — there is nothing to crop away. That is not a
                # reason to skip the frame, which is what this branch used
                # to do: an athlete close enough to fill the shot was
                # reported as "nobody in frame" for as long as they stayed
                # there. Fall through and run the whole frame.
                use_roi = False
            else:
                try:
                    people = self._impl.infer(crop, self.det_confidence)
                except Exception as e:
                    log.error("ROI inference error: %s", e)
                    people = []
                if people:
                    mode = "roi"
                    offset = (ox, oy)
                    scale_src = (crop.shape[1], crop.shape[0])
                else:
                    # The lock came up empty. Rather than reporting "no
                    # athlete" from a crop that may simply have been in
                    # the wrong place, fall through to a full-frame pass
                    # in this same call — the athlete never sees a gap.
                    track.misses += 1
                    if track.misses >= self.MAX_ROI_MISSES:
                        track.clear()
                    mode = "roi->full"
                    use_roi = False

        if not use_roi and not people:
            # The whole frame normally goes to the finder, which is the
            # tracker itself unless the budget pushed the tracker below
            # the size at which a distant athlete can be found at all.
            #
            # The exception is an athlete who already fills the shot —
            # someone doing push-ups a metre from the phone, where there
            # is nothing to crop away. A small model goes blind on *small*
            # people, not on large ones, so there is no reason to pay the
            # finder's cost here; the frame is its own close-up.
            near = (track.box is not None
                    and (track.box[3] - track.box[1]) >= self.TRACKER_OK_BOX_FRAC)
            first = self._impl if near else self._finder
            try:
                people = first.infer(frame_bgr, self.det_confidence)
                if not people and first is not self._finder:
                    # We were wrong about how big they are — don't report
                    # an empty room on the word of the small model.
                    people = self._finder.infer(frame_bgr, self.det_confidence)
            except Exception as e:
                log.error("Inference error: %s", e)
                return None
            track.frames_since_full = 0

            # A full-frame pass at a small input size finds *where* the
            # athlete is very reliably, but its keypoints are coarse:
            # measured against a 640px reference on a 1280x720 frame,
            # full-frame@192 puts joints an average of 83 px out, which
            # would make every elbow angle this app reports meaningless.
            # The same frame re-run on the located crop lands within
            # 14 px. So acquisition costs two passes and tracking costs
            # one — and acquisition is one frame in FULL_RESCAN_EVERY.
            if self.roi_enabled and people and not probe:
                coarse = self._largest(people, frame_bgr.shape[1], frame_bgr.shape[0])
                crop, cox, coy = self._crop(frame_bgr, coarse)
                if crop is not None and crop.size:
                    try:
                        refined = self._impl.infer(crop, self.det_confidence)
                    except Exception:
                        refined = []
                    if refined:
                        people = refined
                        mode = "full+refine"
                        offset = (cox, coy)
                        scale_src = (crop.shape[1], crop.shape[0])

        ms = (time.perf_counter() - t0) * 1000.0

        if not people:
            track.misses += 1
            if track.misses >= self.MAX_ROI_MISSES:
                track.clear()
            return PoseResult(dict(_ZERO_KEYPOINTS), False, 0, "unknown", 0.0, None, ms, 0, mode)

        # Pick the athlete: biggest confident person, with a bonus for
        # overlapping where we saw them last frame (stable tracking when
        # someone walks through the background during a demo).
        cw, ch = scale_src
        ox, oy = offset
        best, best_score = None, -1.0
        for box, score, kpts in people:
            nb = ((float(box[0]) + ox) / w, (float(box[1]) + oy) / h,
                  (float(box[2]) + ox) / w, (float(box[3]) + oy) / h)
            area = max(1e-6, (nb[2] - nb[0]) * (nb[3] - nb[1]))
            s = score * (area ** 0.5)
            if track.box is not None:
                s *= 1.0 + 2.0 * _iou(nb, track.box)
            if s > best_score:
                best, best_score = (nb, score, kpts), s
        nb, score, kpts = best

        keypoints: Dict[str, Tuple[float, float, float]] = {}
        high = 0
        for i, name in enumerate(KEYPOINT_NAMES):
            x = (float(kpts[i, 0]) + ox) / w
            y = (float(kpts[i, 1]) + oy) / h
            c = float(kpts[i, 2])
            keypoints[name] = (x, y, c)
            if c >= self.min_confidence:
                high += 1

        core_confs = [keypoints[j][2] for j in CORE_JOINTS]
        core_visible = sum(1 for c in core_confs if c >= self.min_confidence)
        quality = float(sum(core_confs) / len(core_confs))
        detected = core_visible >= self.min_detected_joints

        # Only a confident detection is allowed to move the ROI lock. A
        # weak one keeps the old box, so a half-second of bad lighting
        # doesn't drag the crop off the athlete.
        if detected:
            track.box = nb
            track.misses = 0
            track.hits += 1
            track.frames_since_full += 1
        else:
            track.misses += 1
            if track.misses >= self.MAX_ROI_MISSES:
                track.clear()

        def side_score(s):
            return sum(keypoints[f"{s}_{j}"][2] for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))

        ls, rs = side_score("left"), side_score("right")
        dominant = "left" if ls > rs and ls > 1.0 else ("right" if rs > ls and rs > 1.0 else "unknown")

        return PoseResult(
            keypoints=keypoints,
            is_detected=detected,
            high_conf_count=high,
            dominant_side=dominant,
            quality=quality,
            box=nb,
            inference_ms=ms,
            people_in_frame=len(people),
            mode=mode,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _largest(people, w: int, h: int) -> Tuple[float, float, float, float]:
        """Normalized box of the most prominent person in a raw result."""
        best, score = people[0][0], -1.0
        for box, conf, _ in people:
            a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            s = conf * (a ** 0.5)
            if s > score:
                best, score = box, s
        return (best[0] / w, best[1] / h, best[2] / w, best[3] / h)

    def _crop(self, frame: np.ndarray, box) -> Tuple[Optional[np.ndarray], int, int]:
        """A padded square crop around the last known box, clipped to frame.

        Square, because the model's input is square: a square crop is the
        one shape that letterboxes to zero padding, so every pixel the
        model looks at is a pixel of the scene.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        span = max(x2 - x1, y2 - y1)
        side = max(span * self.ROI_PAD, 64.0)
        # A square crop cannot be larger than the frame's shorter side.
        # Clamping to it rather than giving up is the point: on a 16:9
        # frame the biggest square still throws away 44% of the picture —
        # all of it wall — so even an athlete who fills the height is
        # better served by a crop than by a letterboxed whole frame,
        # where nearly half the model's input is padding and scenery.
        side = min(side, float(min(w, h)))
        # Unless the athlete is genuinely bigger than that square, in
        # which case cropping would cut them in half. Then the caller
        # falls back to the whole frame, which is the right answer.
        if side < span * 1.05:
            return None, 0, 0
        half = side * 0.5
        ax = int(round(min(max(0.0, cx - half), max(0.0, w - side))))
        ay = int(round(min(max(0.0, cy - half), max(0.0, h - side))))
        bx = int(round(min(float(w), ax + side)))
        by = int(round(min(float(h), ay + side)))
        if bx - ax < 32 or by - ay < 32:
            return None, 0, 0
        return frame[ay:by, ax:bx], ax, ay


def _time_model(impl, probe: np.ndarray, conf: float, runs: int = 8) -> float:
    """Median forward-pass time in ms, warm-up excluded."""
    try:
        for _ in range(2):
            impl.infer(probe, conf)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            impl.infer(probe, conf)
            ts.append((time.perf_counter() - t) * 1000.0)
        return float(np.median(ts))
    except Exception:
        return float("inf")


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ======================================================================
# Backend: ONNX Runtime
# ======================================================================
class _OnnxBackend:
    # More threads than this on a laptop costs more in coordination than
    # it buys on a model this small, and it starves the rest of the app.
    MAX_AUTO_THREADS = 4

    def __init__(self, model_path: str, threads: Optional[int] = None):
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        n = threads or min(os.cpu_count() or 2, self.MAX_AUTO_THREADS)
        so.intra_op_num_threads = int(n)
        so.inter_op_num_threads = 1

        available = ort.get_available_providers()
        preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                 "CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=preferred)
        used = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640

        # Preallocated so the hot path allocates nothing per frame.
        self._canvas = np.empty((self.size, self.size, 3), dtype=np.uint8)
        self._geom: Optional[tuple] = None      # cache of the letterbox maths

        device = {
            "CUDAExecutionProvider": "NVIDIA GPU (CUDA)",
            "DmlExecutionProvider": "GPU (DirectML)",
            "CoreMLExecutionProvider": "Apple Neural Engine (CoreML)",
        }.get(used, f"CPU x{so.intra_op_num_threads}")
        self.info = EngineInfo(
            backend="ONNX Runtime",
            model=_pretty_model_name(model_path),
            device=device,
            input_size=self.size,
            extra={"provider": used, "threads": so.intra_op_num_threads,
                   "file": os.path.basename(model_path)},
        )

    # ------------------------------------------------------------------
    def _letterbox(self, img: np.ndarray):
        """Scale-and-pad into the preallocated square canvas.

        The scale factor and padding depend only on the frame's shape, so
        they are computed once per shape and reused — the camera does not
        change resolution mid-set.
        """
        h, w = img.shape[:2]
        s = self.size
        if self._geom is None or self._geom[0] != (h, w):
            r = min(s / w, s / h)
            nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
            self._geom = ((h, w), r, nw, nh, (s - nw) // 2, (s - nh) // 2)
        _, r, nw, nh, padx, pady = self._geom

        if nw != s or nh != s:
            self._canvas[:] = 114
        cv2.resize(img, (nw, nh), dst=self._canvas[pady:pady + nh, padx:padx + nw],
                   interpolation=cv2.INTER_LINEAR)
        # blobFromImage does BGR->RGB, /255, HWC->NCHW and the float cast
        # in one pass of optimized C++ instead of four numpy temporaries.
        return cv2.dnn.blobFromImage(self._canvas, 1.0 / 255.0, (s, s), swapRB=True), r, padx, pady

    def infer(self, img: np.ndarray, conf_thres: float):
        h, w = img.shape[:2]
        blob, r, padx, pady = self._letterbox(img)

        out = self.session.run(None, {self.input_name: blob})[0][0]   # (56, N)

        # Threshold on the raw score row before any transpose or copy:
        # of 8400 candidates a handful survive, and everything after this
        # line then works on those few columns instead of on all of them.
        scores = out[4]
        idx = np.flatnonzero(scores > conf_thres)
        if idx.size == 0:
            return []
        if idx.size > 32:
            idx = idx[np.argpartition(-scores[idx], 32)[:32]]
        pred = out[:, idx].T                                          # (k, 56)
        sc = scores[idx]

        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes_xywh = np.stack([cx - bw * 0.5, cy - bh * 0.5, bw, bh], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), sc.tolist(), conf_thres, 0.5)
        if len(keep) == 0:
            return []
        keep = np.asarray(keep).reshape(-1)[:5]

        inv = 1.0 / r
        people = []
        for i in keep:
            x1 = (boxes_xywh[i, 0] - padx) * inv
            y1 = (boxes_xywh[i, 1] - pady) * inv
            x2 = x1 + boxes_xywh[i, 2] * inv
            y2 = y1 + boxes_xywh[i, 3] * inv
            k = pred[i, 5:].reshape(17, 3).copy()
            k[:, 0] = (k[:, 0] - padx) * inv
            k[:, 1] = (k[:, 1] - pady) * inv
            people.append(((max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                           float(sc[i]), k))
        return people


# ======================================================================
# Backend: Ultralytics (PyTorch)
# ======================================================================
class _UltralyticsBackend:
    def __init__(self, model_name: str, require_gpu: bool = False):
        import torch  # noqa: F401  (ImportError -> try next backend)
        from ultralytics import YOLO

        cuda = torch.cuda.is_available()
        if require_gpu and not cuda:
            raise RuntimeError("no CUDA GPU")
        self.device = "cuda:0" if cuda else "cpu"
        self.model = YOLO(model_name)
        self.info = EngineInfo(
            backend="Ultralytics + PyTorch",
            model=_pretty_model_name(model_name),
            device=(torch.cuda.get_device_name(0) if cuda else "CPU"),
            input_size=640,
        )

    def infer(self, img: np.ndarray, conf_thres: float):
        res = self.model.predict(img, device=self.device, conf=conf_thres, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kc = res.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2], dtype=np.float32)
        people = []
        for i in range(len(boxes)):
            k = np.concatenate([kxy[i], kc[i][:, None]], axis=1)
            people.append((tuple(boxes[i].tolist()), float(scores[i]), k))
        return people
