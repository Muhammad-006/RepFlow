"""
server.py — RepFlow AI Coach Server.

    python server.py

A backend and nothing else. It takes encrypted camera frames in and sends
coaching JSON back. There is no dashboard, no video stream, no web page,
no browser window and no status console: v2 shipped all of those and they
were a liability, not a feature — a projector-friendly live video feed of
someone exercising is precisely the thing this version promises never to
create.

What it will tell you at startup is one line: that it is listening, and
its identity fingerprint, so you can check the phone is talking to this
machine and not something in between.

API — every body below is an encrypted box (see crypto_box.py), except
the two handshake calls which are by definition pre-key:

    GET  /v1/hello              offer: identity key, fresh ephemeral key,
                                exercise catalog. Unencrypted by necessity.
    POST /v1/session            {hs, pub} -> sealed {session_id}
    POST /v1/set                sealed {exercise, target} -> sealed {ok}
    POST /v1/frame              sealed (header + JPEG) -> sealed result
    GET  /v1/summary            -> sealed end-of-set analytics
    POST /v1/end                -> sealed privacy receipt, then keys wiped
    GET  /v1/health             {"status":"ok"} — for discovery only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

import crypto_box as cb
import exercises
from discovery import DISCOVERY_PORT, DiscoveryService
from pose_engine import PoseEngine
from session import ROTATIONS, ClockSync, Session

VERSION = "3.2.0"
HERE = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("repflow")
logging.getLogger("repflow.discovery").setLevel(logging.WARNING)

# ----------------------------------------------------------------------
# Where the server's long-term identity key lives.
#
# This used to be a dotfile next to server.py (inside the app folder).
# That is exactly wrong: every time the app folder is replaced — a new
# zip extracted, a fresh `git clone`, `git pull` onto a clean checkout —
# the key vanished, `load_or_create_identity` generated a brand new one,
# and every phone that had ever paired with this machine saw "this
# coach's identity has changed" the next time it connected. The identity
# check was doing exactly its job (a different key really did answer);
# the file just shouldn't have been something a code update could delete.
#
# `client/discovery.py` already solved this for the phone's own state —
# `~/.repflow/repflow_state.json`, outside the app bundle — so the server
# now uses the same directory for the same reason. One key per machine,
# survives every code update, and a fresh app folder never invalidates a
# pairing that already exists.
#
# A legacy key sitting next to server.py (from before this fix) is
# migrated in rather than discarded, so upgrading to this version does
# not itself force a re-pair.
IDENTITY_MIGRATED = False


def _default_identity_path() -> str:
    global IDENTITY_MIGRATED
    new_dir = os.path.join(os.path.expanduser("~"), ".repflow")
    new_path = os.path.join(new_dir, "server_identity")
    legacy_path = os.path.join(HERE, ".repflow_identity")
    try:
        os.makedirs(new_dir, exist_ok=True)
        if not os.path.isfile(new_path) and os.path.isfile(legacy_path):
            shutil.copy2(legacy_path, new_path)
            try:
                os.chmod(new_path, 0o600)
            except OSError:
                pass
            IDENTITY_MIGRATED = True
            log.info("migrated identity key from %s to %s", legacy_path, new_path)
        return new_path
    except OSError as e:
        # A locked-down machine, a read-only home directory — rare, but
        # the server should still start rather than crash on this.
        log.warning("could not use %s (%s); falling back to %s — this key will "
                    "NOT survive a fresh copy of this app folder", new_dir, e, legacy_path)
        return legacy_path


IDENTITY_PATH = _default_identity_path()

# A frame this much later than the best recent frame describes a body
# position the athlete has already left. Running the model on it wastes
# the very milliseconds that made it late, so it is dropped and the phone
# is told to send a fresh one.
#
# This is measured against a per-session clock estimate (session.ClockSync),
# NOT by subtracting the phone's wall clock from this machine's. Those two
# clocks disagree by seconds on ordinary hardware, and comparing them
# directly is what made v3.0 silently drop every single frame.
MAX_FRAME_LATENESS_S = ClockSync.LATE_S

# However far behind we fall, never drop more than this many frames in a
# row: a burst of dropped frames must degrade the coaching, not switch it
# off. If the run of drops gets this long, the next frame is processed
# regardless of age so the athlete sees tracking resume.
#
# This was 8. Eight is fine at ten frames a second (under a second of
# degraded coaching) and badly wrong at one frame a second, where it is
# EIGHT SECONDS of a completely blank skeleton while the athlete stares
# at the screen wondering whether the app has crashed — which is exactly
# what got reported. Two is the honest limit: long enough to skip a
# genuine hiccup, short enough that nobody watches a dead screen.
MAX_CONSECUTIVE_DROPS = 2

HANDSHAKE_TTL_S = 120

# A defensive cap, not a quality setting. The benchmarked cost of decode +
# rotate + inference stays under ~20 ms all the way up to 1080p on modest
# hardware, so this is not where an ordinary phone camera's latency comes
# from — it exists only so that an unusually large capture (a newer phone
# defaulting to a bigger preset, a future camera plugin) can never turn
# into an unbounded per-frame cost. Longest edge, aspect ratio preserved.
MAX_FRAME_EDGE = 1280

# Socket writes/reads slower than this are flagged in the log as a
# network/Wi-Fi symptom rather than a server-compute one — see the
# Handler.disable_nagle_algorithm comment for why server-side timing
# alone can't see a Nagle/delayed-ACK stall, but CAN see a genuinely slow
# or congested link (the write/read call itself blocks longer).
SLOW_WRITE_WARN_MS = 150.0
SLOW_READ_WARN_MS = 150.0

# How often, in frames, to print the latency breakdown during a live set.
# The point of this is to end the guessing. The phone shows one number —
# a full round trip — and three rounds of fixes have now been argued from
# that single number without knowing which leg of the trip owned it. The
# server can see three of the four legs directly (body read, its own
# compute, reply write) and can therefore derive the fourth (everything
# happening on the phone: capture, JPEG encode, crypto, Flet's own
# plumbing). That derived number is the one that decides whether there is
# anything left to fix on this side at all.
LATENCY_REPORT_EVERY = 40

_engine: Optional[PoseEngine] = None
_identity_priv: bytes = b""
_identity_pub: bytes = b""
_sessions: Dict[str, Session] = {}
_sessions_lock = threading.Lock()
_pending: Dict[str, Tuple[bytes, float]] = {}   # handshake id -> (ephemeral priv, created)
_pending_lock = threading.Lock()
_started = time.time()

# ----------------------------------------------------------------------
# Connections that vanish mid-request are normal operation, not faults.
# A phone whose screen locks, an app the athlete swipes away, a Wi-Fi
# roam between access points, and above all the discovery probe — which
# opens a socket, reads one health reply and hangs up immediately — all
# abort a TCP connection without a polite close. Python's
# ThreadingHTTPServer answers each of those by printing a full traceback
# (`ConnectionResetError: [WinError 10054]` on Windows, ECONNRESET
# elsewhere) from inside a worker thread. Nothing is wrong, nothing is
# actionable, and at 10 frames a second the real log lines — session
# opened, keys destroyed — drown in it.
#
# So they are caught, counted and summarised instead. A rising count is
# still visible with --verbose, which is the only case where it means
# anything: many drops with no sessions opening is a firewall eating the
# handshake, and that is worth seeing.
DROPPED_CONNECTION_ERRORS = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,          # socket.timeout is an alias of this on 3.10+
    socket.timeout,
)
_dropped_conns = 0
_dropped_lock = threading.Lock()


def _note_dropped(exc: BaseException, where: str = "") -> None:
    global _dropped_conns
    with _dropped_lock:
        _dropped_conns += 1
        n = _dropped_conns
    log.debug("client connection dropped%s (%s) [%d so far]",
              f" during {where}" if where else "", type(exc).__name__, n)


# ======================================================================
# handshake
# ======================================================================
def new_handshake() -> Tuple[str, bytes]:
    priv, pub = cb.generate_keypair()
    hs = secrets.token_hex(8)
    now = time.time()
    with _pending_lock:
        for k, (_, t) in list(_pending.items()):
            if now - t > HANDSHAKE_TTL_S:
                del _pending[k]
        _pending[hs] = (priv, now)
    return hs, pub


def complete_handshake(hs: str, client_pub: bytes) -> Optional[dict]:
    with _pending_lock:
        entry = _pending.pop(hs, None)
    if entry is None:
        return None
    eph_priv, created = entry
    if time.time() - created > HANDSHAKE_TTL_S:
        return None
    try:
        return cb.server_handshake(_identity_priv, eph_priv, client_pub)
    except cb.CryptoError:
        return None


# ======================================================================
# frame processing
# ======================================================================
def _rotate(img, r: int):
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _norm_rot(v, default=0) -> int:
    try:
        r = int(round(float(v) / 90.0)) * 90 % 360
        return r if r in ROTATIONS else default
    except (TypeError, ValueError):
        return default


def process_frame(session: Session, jpeg: memoryview, hint: int, display_rot: int,
                  capture_age_s: float) -> dict:
    """Decode → pose → coach. The decoded image never leaves this function."""
    t0 = time.perf_counter()
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "undecodable frame"}
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > MAX_FRAME_EDGE:
        scale = MAX_FRAME_EDGE / float(longest)
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)

    best = None
    cands = session.orient.candidates(hint)
    for r in cands:
        img = _rotate(frame, r)
        # The ROI lock belongs to one rotation. Probing a different one
        # must not be allowed to disturb it, or a single probe frame
        # would throw away a perfectly good lock.
        own = session.prev_rot == r
        track = session.track if own else None
        # Every candidate after the first is a rotation probe: it only
        # has to answer "is anyone there", so it skips the refinement
        # pass and costs half as much.
        pose = _engine.process_frame(img, track=track, probe=not own and len(cands) > 1)
        if pose is None:
            continue
        session.orient.observe(r, pose.is_detected, pose.quality)
        score = pose.quality if pose.is_detected else -1.0
        if best is None or score > best[0]:
            best = (score, r, img.shape[1] / float(img.shape[0]), pose)
        del img
    del frame

    if best is None:
        return {"error": "inference failed"}
    _, rot, aspect, pose = best
    if rot != session.prev_rot:
        # The winning rotation changed, so the old lock describes a frame
        # that no longer exists. Re-seed it from what this rotation found.
        session.track.clear()
        if pose.is_detected and pose.box is not None:
            session.track.box, session.track.hits = pose.box, 1
        session.prev_rot = rot
    return session.update(pose, rot, display_rot, aspect, capture_age_s,
                          (time.perf_counter() - t0) * 1000.0)


# ======================================================================
# HTTP
# ======================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = f"RepFlow/{VERSION}"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    # Nagle's algorithm + delayed ACK. socketserver leaves Nagle ON by
    # default (`disable_nagle_algorithm = False`), and it is a documented
    # Python gotcha: a small header write followed by a separate body
    # write can sit waiting on the peer's delayed ACK before the second
    # segment goes out — commonly 40 ms on Linux, up to ~200 ms on
    # Windows, per request. At 10 fps that alone can eat a whole frame
    # budget, and it does not show up in server-side timing at all,
    # because the write() call itself returns immediately — the stall
    # happens in the kernel/network layer after control returns to
    # Python. This is why the "700 ms" a phone sees can coexist with a
    # server that benchmarks at ~15-20 ms of actual compute.
    #
    # Two independent fixes, both applied: disable Nagle on this socket
    # (below), and stop sending headers and body as two writes (_send).
    disable_nagle_algorithm = True

    def log_message(self, fmt, *args):
        pass       # per-request logging at 10 fps is noise, and it is metadata

    def handle_one_request(self):
        """As the base class, but a hung-up client is not an exception.

        This is where the traceback used to come from: the socket dies
        while the header line is being read, and there is no answer to
        send to a peer that has already gone.
        """
        try:
            super().handle_one_request()
        except DROPPED_CONNECTION_ERRORS as e:
            self.close_connection = True
            _note_dropped(e, "request")

    # -- plumbing ------------------------------------------------------
    # Set by _send on every reply so do_POST can report the write cost as
    # part of the per-frame latency breakdown.
    _last_write_ms = 0.0
    _last_reply_bytes = 0

    def _send(self, status: int, body: bytes, ctype: str):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Deliberately NOT self.end_headers() + self.wfile.write(body)
            # (two socket writes). Everything queued in the header buffer
            # plus the blank line plus the body goes out in ONE write, so
            # there is nothing left for Nagle to hold back waiting on.
            if self.request_version != "HTTP/0.9":
                self._headers_buffer.append(b"\r\n")
            payload = b"".join(self._headers_buffer) + body
            self._headers_buffer = []
            t0 = time.perf_counter()
            self.wfile.write(payload)
            write_ms = (time.perf_counter() - t0) * 1000.0
            self._last_write_ms = write_ms
            self._last_reply_bytes = len(payload)
            if write_ms > SLOW_WRITE_WARN_MS:
                log.warning("slow reply write: %.0f ms for %d bytes — this is "
                           "network/Wi-Fi time (or a congested/weak signal), "
                           "not server compute", write_ms, len(payload))
        except DROPPED_CONNECTION_ERRORS as e:
            # The phone stopped listening between asking and being
            # answered. The work is already done and the reply is worth
            # nothing now; drop the connection and carry on.
            self.close_connection = True
            _note_dropped(e, "reply")

    def _json(self, status: int, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _session(self) -> Optional[Session]:
        sid = self.headers.get("X-Session-Id") or ""
        with _sessions_lock:
            return _sessions.get(sid)

    def _sealed(self, session: Session, status: int, payload: dict, aad: bytes = None):
        """Encrypt a JSON reply for this phone and nobody else.

        `aad` defaults to the session id, which binds every reply to the
        session it belongs to. The one exception is the reply that *hands
        out* the session id — the phone cannot bind to an id it has not
        been told yet, so that single message binds to nothing extra. It
        is still sealed with the freshly agreed keys, which is exactly
        what makes it proof that the server is genuine.
        """
        session._out_counter = getattr(session, "_out_counter", 0) + 1
        box = cb.seal(session.key("s2c_enc"), session.key("s2c_mac"),
                      session._out_counter, json.dumps(payload).encode("utf-8"),
                      aad=session.id.encode("ascii") if aad is None else aad)
        self._send(status, box, "application/octet-stream")

    def _open(self, session: Session, body: bytes, path: str) -> Optional[bytes]:
        """Authenticate, replay-check and decrypt an incoming box."""
        try:
            counter, plaintext = cb.open_box(
                session.key("c2s_enc"), session.key("c2s_mac"), body,
                aad=(session.id + path).encode("ascii"))
        except cb.CryptoError:
            session.privacy.frames_rejected += 1
            return None
        if not session.replay.check_and_mark(counter):
            session.privacy.frames_rejected += 1
            return None
        return plaintext

    # -- routes --------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/v1/health":
            # Everything discovery needs and nothing it doesn't: no
            # hostname, no OS, no local addresses, no session list.
            return self._json(200, {"service": "repflow", "status": "ok",
                                    "version": VERSION, "ready": _engine is not None})

        if path == "/v1/hello":
            hs, eph_pub = new_handshake()
            return self._json(200, {
                "version": VERSION,
                "protocol": cb.PROTOCOL.decode(),
                "hs": hs,
                "identity_pub": _identity_pub.hex(),
                "ephemeral_pub": eph_pub.hex(),
                "fingerprint": cb.fingerprint(_identity_pub),
                "safety_words": cb.fingerprint_words(_identity_pub),
                "exercises": exercises.catalog(),
                "roadmap": exercises.ROADMAP,
            })

        if path == "/v1/summary":
            s = self._session()
            if s is None:
                return self._json(404, {"error": "no session"})
            with s.lock:
                return self._sealed(s, 200, s.summary())

        return self._json(404, {"error": "not found"})

    def _report_latency(self, s, result: dict, read_ms: float, req_bytes: int):
        """Print, every LATENCY_REPORT_EVERY frames, where the round trip
        the phone displays is actually being spent.

        The arithmetic is the useful part. With one frame in flight at a
        time, the gap between two frames arriving here IS the round trip
        the phone's latency chip shows. This server can time three of its
        four legs; whatever is left over happened on the phone or in the
        air. Printing the leftover explicitly is what stops the next
        round of this being another guess.
        """
        try:
            if s.frames % LATENCY_REPORT_EVERY:
                return
            interval_ms = s.clock.interval_s * 1000.0
            if interval_ms <= 0:
                return
            server_ms = float(result.get("server_ms") or 0.0)
            write_ms = self._last_write_ms
            elsewhere = interval_ms - server_ms - read_ms - write_ms
            fps = 1000.0 / interval_ms
            log.info(
                "latency %.0f ms/frame (%.1f fps) = server %.0f + upload read %.0f "
                "+ reply write %.0f + phone/network %.0f | frame %d B, reply %d B",
                interval_ms, fps, server_ms, read_ms, write_ms, max(0.0, elsewhere),
                req_bytes, self._last_reply_bytes)
            if elsewhere > 250.0 and elsewhere > 4 * server_ms:
                log.warning(
                    "  ^ %.0f ms of that is OFF this server (phone capture + JPEG "
                    "encode + Wi-Fi). Server compute is %.0f ms. Nothing in "
                    "server.py can recover that time — it needs the client-side "
                    "fix (camera preset + one-frame-in-flight) to come down.",
                    elsewhere, server_ms)
        except Exception:      # diagnostics must never break a frame
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        t_read0 = time.perf_counter()
        body = self._body()
        read_ms = (time.perf_counter() - t_read0) * 1000.0
        if path == "/v1/frame" and read_ms > SLOW_READ_WARN_MS:
            log.warning("slow request read: %.0f ms for %d bytes — this is the "
                       "phone's upload over the network/Wi-Fi, not server compute",
                       read_ms, len(body))

        # ---- key agreement ------------------------------------------
        if path == "/v1/session":
            try:
                req = json.loads(body or b"{}")
                client_pub = bytes.fromhex(req["pub"])
                hs = str(req["hs"])
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": "bad handshake"})
            if len(client_pub) != 32:
                return self._json(400, {"error": "bad key"})
            result = complete_handshake(hs, client_pub)
            if result is None:
                return self._json(400, {"error": "handshake expired"})
            sid = secrets.token_hex(16)
            s = Session(sid, result["keys"])
            with _sessions_lock:
                _sessions[sid] = s
            log.info("session %s opened", s.short_id)
            # The reply is encrypted with the freshly derived keys, so a
            # phone that can read it has provably reached the real server.
            return self._sealed(s, 200, {"session_id": sid, "version": VERSION}, aad=b"")

        # ---- everything else is sealed ------------------------------
        s = self._session()
        if s is None:
            return self._json(401, {"error": "no session"})
        plaintext = self._open(s, body, path)
        if plaintext is None:
            return self._json(401, {"error": "authentication failed"})

        if path == "/v1/set":
            try:
                req = json.loads(plaintext or b"{}")
            except ValueError:
                req = {}
            with s.lock:
                s.reset(exercise=req.get("exercise"), target=int(req.get("target", 0) or 0))
            return self._sealed(s, 200, {"ok": True, "exercise": s.exercise_key,
                                         "unit": s.coach.UNIT})

        if path == "/v1/frame":
            t_recv = time.time()
            s.privacy.frames_received += 1
            s.privacy.bytes_received += len(body)
            try:
                hdr_len = int.from_bytes(plaintext[:2], "big")
                header = json.loads(plaintext[2:2 + hdr_len])
                jpeg = memoryview(plaintext)[2 + hdr_len:]
            except (ValueError, IndexError):
                return self._sealed(s, 200, {"error": "malformed payload"})
            s.privacy.frames_decrypted += 1

            # How late is this frame *relative to this phone's own best
            # frame*? Anything absolute would be measuring clock skew.
            try:
                t_sent = float(header.get("t", t_recv))
            except (TypeError, ValueError):
                t_sent = t_recv
            with s.lock:
                age = s.clock.observe(t_recv, t_sent)
                # Three reasons never to drop, all of them "dropping this
                # frame cannot possibly help":
                #  - the phone is round-trip-bound, so there is no fresher
                #    frame behind this one to skip ahead to (see
                #    ClockSync.round_trip_bound);
                #  - we have already skipped our run of drops;
                #  - the exercise is a hold, where a body that is
                #    deliberately not moving is described just as well by
                #    a frame that is a fraction of a second old, and the
                #    alternative is a plank timer that stops counting.
                is_hold = s.coach.UNIT == "seconds"
                enforce = (s.clock.ready
                           and s.stale_drops < MAX_CONSECUTIVE_DROPS
                           and not s.clock.round_trip_bound
                           and not is_hold)
                stale = enforce and age > MAX_FRAME_LATENESS_S
                if stale:
                    s.stale_drops += 1
                    s.last_drop_reason = "late"
                else:
                    s.stale_drops = 0
            if stale:
                # Too late to coach from. Say so rather than answering late.
                return self._sealed(s, 200, {"stale": True, "age_ms": round(age * 1000),
                                             "count": s.last.get("count", 0),
                                             "is_detected": s.last.get("is_detected", False)})
            try:
                with s.lock:
                    result = process_frame(s, jpeg, _norm_rot(header.get("r"), 0),
                                           _norm_rot(header.get("d"), 0), age)
            except Exception:
                log.exception("frame processing failed (session %s)", s.short_id)
                result = {"error": "internal error"}
            finally:
                # The decoded image and the decrypted JPEG both die here.
                jpeg.release()
                s.privacy.plaintext_lifetime_ms += (time.time() - t_recv) * 1000.0
                del plaintext
            out = self._sealed(s, 200, result)
            self._report_latency(s, result, read_ms, len(body))
            return out

        if path == "/v1/end":
            with s.lock:
                summary = s.summary()
                receipt = summary.get("privacy", {})
            with _sessions_lock:
                _sessions.pop(s.id, None)
            payload = {"summary": summary, "receipt": receipt}
            self._sealed(s, 200, payload)
            s.destroy()      # keys zeroed only after the reply is on the wire
            log.info("session %s closed and keys destroyed", s.short_id)
            return

        return self._json(404, {"error": "not found"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        """The last line of defence against traceback spam.

        Anything that got past the handler's own guard and is just a
        connection going away is counted and swallowed; a real bug still
        prints in full, because a silent server is worse than a noisy one.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, DROPPED_CONNECTION_ERRORS):
            _note_dropped(exc, "connection")
            return
        log.exception("unhandled error serving %s", client_address)


# ======================================================================
def _reaper():
    """Drop idle sessions and wipe their keys. Nothing is kept 'just in case'."""
    while True:
        time.sleep(30)
        with _sessions_lock:
            stale = [k for k, s in _sessions.items() if s.expired]
            for k in stale:
                _sessions.pop(k).destroy()
        if stale:
            log.info("reaped %d idle session(s)", len(stale))


def main():
    global _engine, _identity_priv, _identity_pub

    ap = argparse.ArgumentParser(description="RepFlow AI Coach Server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "ultralytics"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for ONNX Runtime (raise this if inference is slow)")
    ap.add_argument("--imgsz", default="auto",
                    help="pose model input size: auto (benchmark and pick), or 192/256/320/640")
    ap.add_argument("--latency-budget", type=float, default=8.0, metavar="MS",
                    help="target milliseconds per forward pass; --imgsz auto picks the "
                         "largest model that fits (default: 8)")
    ap.add_argument("--no-roi", action="store_true",
                    help="disable athlete-crop tracking (always run on the whole frame)")
    ap.add_argument("--no-discovery", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--identity-path", default=None, metavar="FILE",
                    help="override where the identity key lives (default: "
                         "~/.repflow/server_identity — a fresh copy of this app "
                         "folder keeps using the same key from here on out)")
    args = ap.parse_args()

    if not args.verbose:
        logging.getLogger("repflow.pose").setLevel(logging.WARNING)

    identity_path = (os.path.abspath(os.path.expanduser(args.identity_path))
                     if args.identity_path else IDENTITY_PATH)
    _identity_priv = cb.load_or_create_identity(identity_path)
    _identity_pub = cb.public_from_private(_identity_priv)

    _engine = PoseEngine(backend=args.backend, model=args.model, threads=args.threads,
                         imgsz=args.imgsz, budget_ms=args.latency_budget,
                         roi=not args.no_roi)

    try:
        httpd = _Server((args.host, args.port), Handler)
    except OSError as e:
        print(f"Port {args.port} is not available ({e}). Try --port 8770.", file=sys.stderr)
        sys.exit(1)

    threading.Thread(target=_reaper, name="reaper", daemon=True).start()
    if not args.no_discovery:
        DiscoveryService(args.port, lambda: {
            "v": VERSION,
            "fp": cb.fingerprint(_identity_pub),
        }).start()

    # The entire startup output: what is running, how fast it measured on
    # this machine, and the words the user compares against their phone.
    info = _engine.info
    mark = "✓" if info.within_budget else "!"
    print(f"RepFlow coach server {VERSION} listening on :{args.port}  "
          f"({info.model} · {info.backend} · {info.device})")
    print("Nagle's algorithm disabled on replies + headers/body sent in one write "
          "(fixes a documented 40-200 ms Python HTTP stall per frame)")
    print("Low-frame-rate coaching fixes active: calibration, the plank clock and "
          "the wobble window are measured in real seconds, and late frames are "
          "coached from rather than dropped whenever the phone is round-trip-bound")
    print("A latency breakdown prints every "
          f"{LATENCY_REPORT_EVERY} frames during a set — it says which leg of the "
          "round trip owns the time")
    print(f"Tracking: {info.input_size}px on the athlete, {info.bench_ms:.1f} ms "
          f"measured here  [{mark} budget {info.budget_ms:.0f} ms]")
    if info.finder_size and info.finder_size != info.input_size:
        # Two numbers, because there are two jobs. Showing only the fast
        # one would be the same self-flattery that shipped a 192px model
        # that could not see anyone standing more than a metre away.
        print(f"Finding:  {info.finder_size}px on the whole frame, "
              f"{info.finder_ms:.1f} ms — while lost, and once every "
              f"{PoseEngine.FULL_RESCAN_EVERY} frames while tracking")
    if not info.within_budget:
        print("  ↳ over budget: try --threads "
              f"{min(8, (os.cpu_count() or 4))} to use more cores, or raise "
              "--latency-budget.")
    print(f"Safety words: {cb.fingerprint_words(_identity_pub)}")
    print(f"Identity: {identity_path}" + ("  (migrated from the app folder)"
                                          if IDENTITY_MIGRATED else ""))
    if identity_path == os.path.join(HERE, ".repflow_identity"):
        print("  ↳ this key lives INSIDE the app folder — a fresh copy of this app "
              "(a new zip, a fresh\n    clone) will generate a new one and every "
              "paired phone will need to re-pair.\n    Move it with --identity-path, "
              "or fix the permissions on your home folder so the\n    default "
              "location can be used.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with _sessions_lock:
            for s in _sessions.values():
                s.destroy()
            _sessions.clear()
        httpd.server_close()


if __name__ == "__main__":
    main()
