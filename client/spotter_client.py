"""
spotter_client.py — the phone's end of the encrypted link.

Standard library only (`http.client`, `json`, `struct`, plus the project's
own `crypto_box`), because everything here has to survive being compiled
into an Android APK, and no native crypto package builds for Android.

Every method blocks, so call them from a worker thread
(`page.run_thread`), never from Flet's event loop.

The shape of a session:

    connect()      X25519 handshake → four session keys the server and
                   this phone both hold and nobody else can derive
    start_set()    tell the server which exercise and what the goal is
    send_frame()   one sealed JPEG → one sealed coaching result
    finish()       fetch the report, then tell the server to destroy its
                   copy of the keys

Frames are sealed with a strictly increasing counter, and the server
refuses a counter it has seen before, so a frame captured off the network
cannot be replayed back at it.
"""

from __future__ import annotations

import http.client
import json
import secrets
import socket
import ssl
import struct
import threading
import time
from typing import Optional, Tuple

import crypto_box as cb


class SpotterClientError(Exception):
    """Any failure talking to the coach server."""


class SecurityError(SpotterClientError):
    """The server is not the one we expected, or a reply failed to verify."""


class _Secure:
    """Session state shared by every clone of a client.

    `http.client` connections are not thread-safe, so the live screen and
    the summary loader each get their own socket — but they must share one
    session, one set of keys and one counter, or the server will see a
    replayed counter and start rejecting frames.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.keys: Optional[dict] = None
        self.session_id: Optional[str] = None
        self.counter = 0
        self.salt = b""
        self.fingerprint = ""
        self.safety_words = ""
        self.catalog: list = []
        self.roadmap: list = []
        self.server_version = ""

    def next_counter(self) -> int:
        with self.lock:
            self.counter += 1
            return self.counter

    def clear(self):
        with self.lock:
            self.keys = None
            self.session_id = None
            self.counter = 0


class SpotterClient:
    def __init__(self, host: str = "", port: int = 8765, timeout: float = 4.0,
                 use_https: bool = False, secure: Optional[_Secure] = None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.use_https = use_https
        self.secure = secure or _Secure()
        self._conn: Optional[http.client.HTTPConnection] = None
        self.last_rtt_ms: Optional[float] = None
        self.last_server_ms: Optional[float] = None

    # ------------------------------------------------------------------
    def clone(self) -> "SpotterClient":
        return SpotterClient(self.host, self.port, self.timeout, self.use_https, self.secure)

    @property
    def configured(self) -> bool:
        return bool(self.host)

    @property
    def connected(self) -> bool:
        return bool(self.secure.session_id and self.secure.keys)

    @property
    def safety_words(self) -> str:
        return self.secure.safety_words

    @property
    def fingerprint(self) -> str:
        return self.secure.fingerprint

    def retarget(self, host: str, port: int = 8765, use_https: Optional[bool] = None):
        https = self.use_https if use_https is None else use_https
        if host != self.host or port != self.port or https != self.use_https:
            self.close()
            self.secure.clear()
        self.host, self.port, self.use_https = host, port, https

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    def _connection(self):
        if not self.host:
            raise SpotterClientError("no server selected")
        if self._conn is None:
            if self.use_https:
                self._conn = http.client.HTTPSConnection(
                    self.host, self.port, timeout=self.timeout,
                    context=ssl.create_default_context())
            else:
                self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn

    def _raw(self, method: str, path: str, body: bytes = b"",
             headers: Optional[dict] = None) -> Tuple[int, bytes]:
        t0 = time.perf_counter()
        try:
            conn = self._connection()
            h = dict(headers or {})
            if self.secure.session_id:
                h["X-Session-Id"] = self.secure.session_id
            conn.request(method, path, body=body, headers=h)
            resp = conn.getresponse()
            data = resp.read()
            self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
            return resp.status, data
        except (OSError, http.client.HTTPException, socket.timeout, ssl.SSLError, ValueError) as e:
            self.close()      # the next call opens a fresh socket: free retry
            raise SpotterClientError(str(e)) from e

    # ------------------------------------------------------------------
    def hello(self) -> dict:
        """The one unencrypted call: ask the server to introduce itself."""
        status, data = self._raw("GET", "/v1/hello")
        if status != 200:
            raise SpotterClientError(f"/v1/hello -> HTTP {status}")
        try:
            return json.loads(data.decode("utf-8"))
        except ValueError as e:
            raise SpotterClientError("malformed hello") from e

    def health(self) -> Optional[dict]:
        try:
            status, data = self._raw("GET", "/v1/health")
            return json.loads(data.decode("utf-8")) if status == 200 else None
        except (SpotterClientError, ValueError):
            return None

    # ------------------------------------------------------------------
    def connect(self, expect_fingerprint: str = "") -> dict:
        """Run the handshake and open a session.

        If `expect_fingerprint` is given (we have talked to this coach
        before), a mismatch raises `SecurityError` rather than connecting.
        That is the moment a machine-in-the-middle would be caught: it can
        relay bytes all day, but it cannot produce the identity key whose
        fingerprint the phone remembered.
        """
        offer = self.hello()
        identity = bytes.fromhex(offer.get("identity_pub", ""))
        ephemeral = bytes.fromhex(offer.get("ephemeral_pub", ""))
        if len(identity) != 32 or len(ephemeral) != 32:
            raise SecurityError("server sent a malformed key")

        fp = cb.fingerprint(identity)
        if expect_fingerprint and fp != expect_fingerprint:
            raise SecurityError("this coach's identity has changed")

        hs = cb.client_handshake(identity, ephemeral)
        body = json.dumps({"hs": offer.get("hs", ""),
                           "pub": hs["client_pub"].hex()}).encode("utf-8")
        status, data = self._raw("POST", "/v1/session", body=body, headers={
            "Content-Type": "application/json", "Content-Length": str(len(body))})
        if status != 200:
            raise SpotterClientError(f"/v1/session -> HTTP {status}")

        keys = hs["keys"]
        try:
            # Only the real server could seal this, so decrypting it is the
            # proof that the handshake reached the machine we pinned.
            _, plaintext = cb.open_box(keys["s2c_enc"], keys["s2c_mac"], data, aad=b"")
            reply = json.loads(plaintext.decode("utf-8"))
        except (cb.CryptoError, ValueError) as e:
            raise SecurityError("could not verify the server's reply") from e

        s = self.secure
        with s.lock:
            s.keys = keys
            s.session_id = reply["session_id"]
            s.counter = 0
            s.salt = secrets.token_bytes(4)
            s.fingerprint = fp
            s.safety_words = offer.get("safety_words", cb.fingerprint_words(identity))
            s.catalog = offer.get("exercises", [])
            s.roadmap = offer.get("roadmap", [])
            s.server_version = offer.get("version", "")
        return offer

    # ------------------------------------------------------------------
    def _sealed(self, path: str, payload: bytes) -> dict:
        s = self.secure
        if not s.keys or not s.session_id:
            raise SpotterClientError("not connected")
        aad = (s.session_id + path).encode("ascii")
        box = cb.seal(s.keys["c2s_enc"], s.keys["c2s_mac"], s.next_counter(),
                      payload, aad=aad, salt=s.salt)
        status, data = self._raw("POST", path, body=box, headers={
            "Content-Type": "application/octet-stream", "Content-Length": str(len(box))})
        if status == 401:
            s.clear()
            raise SecurityError("session rejected — reconnecting")
        if status != 200:
            raise SpotterClientError(f"{path} -> HTTP {status}")
        return self._unseal(data)

    def _unseal(self, data: bytes) -> dict:
        s = self.secure
        try:
            _, plaintext = cb.open_box(s.keys["s2c_enc"], s.keys["s2c_mac"], data,
                                       aad=s.session_id.encode("ascii"))
            result = json.loads(plaintext.decode("utf-8"))
        except (cb.CryptoError, ValueError, AttributeError) as e:
            raise SecurityError("a reply failed to verify") from e
        self.last_server_ms = result.get("server_ms")
        return result

    # ------------------------------------------------------------------
    def start_set(self, exercise: str = "pushup", target: int = 0) -> dict:
        return self._sealed("/v1/set", json.dumps(
            {"exercise": exercise, "target": int(target or 0)}).encode("utf-8"))

    def send_frame(self, jpeg: bytes, rotation_hint: int = 0, display_rotation: int = 0,
                   captured_at: Optional[float] = None) -> dict:
        """Seal one camera frame.

        The rotation hints and the capture timestamp travel *inside* the
        encrypted payload rather than in HTTP headers. Anyone watching the
        network sees a POST of opaque bytes and nothing else — not how the
        phone is propped, not when the shutter fired.
        """
        header = json.dumps({
            "r": int(rotation_hint), "d": int(display_rotation),
            "t": captured_at if captured_at is not None else time.time(),
        }).encode("utf-8")
        payload = struct.pack(">H", len(header)) + header + jpeg
        return self._sealed("/v1/frame", payload)

    def summary(self) -> Optional[dict]:
        s = self.secure
        if not s.keys or not s.session_id:
            return None
        try:
            status, data = self._raw("GET", "/v1/summary")
            if status != 200:
                return None
            return self._unseal(data)
        except SpotterClientError:
            return None

    def finish(self) -> Optional[dict]:
        """Fetch the report and have the server destroy its keys."""
        try:
            result = self._sealed("/v1/end", b"{}")
        except SpotterClientError:
            self.secure.clear()
            return None
        self.secure.clear()
        return result
