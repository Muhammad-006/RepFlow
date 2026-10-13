"""
connection.py — keeps the app linked to the coach server, securely.

Four jobs:

* **Find it.** Discovery runs in the background (discovery.py): same
  Wi-Fi, laptop hotspot, phone hotspot or USB. Nobody types an address —
  unless they choose to (see "override" below).

* **Pick the best one, and don't get stuck on a bad one.** Discovery
  ranks every reachable coach by measured round-trip latency and hands
  back the whole ordered list, nearest first. This class walks that list
  as a strict failover cascade: try the nearest node, and if its health
  check already looked fine but the actual handshake fails or times out,
  fail over *instantly* to the next-ranked node instead of giving up or
  re-running discovery from scratch. See EDGE_ROUTING.md.

* **Verify it.** The first time this phone meets a coach, it remembers
  that coach's identity fingerprint (trust on first use). Every later
  connection is checked against the remembered one, and a mismatch stops
  the *whole cascade* instead of quietly trying a different node — an
  identity mismatch is a security event, not a routing hiccup, and it's
  surfaced to the person rather than silently worked around.

* **Keep it.** A light heartbeat notices the server going away; during a
  set the frames themselves are the heartbeat, so nothing extra is sent.

**Manual override.** A person can force a direct connection to a specific
IP address or WebSocket/HTTP(S) URI from the network panel in the UI,
bypassing auto-discovery entirely. This does not relax anything: the same
strict-protocol check (`discovery.is_private_host`) and the same
trust-on-first-use identity pinning apply to a manually entered address
exactly as they do to an auto-discovered one. The only thing that changes
is *which* address gets tried — never how carefully it's verified.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import config
import discovery
from spotter_client import SecurityError, SpotterClient, SpotterClientError


class ConnectionManager:
    def __init__(self, client: SpotterClient):
        self.client = client
        self.info: Optional[dict] = None
        self.state = "idle"              # idle | searching | online | offline | untrusted
        self.status_text = "Starting…"
        self.rtt_ms: Optional[float] = None
        self.security_warning = ""
        self._listeners: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._searching = False
        self._stop = threading.Event()
        self._live_mode = False
        self.last_ok = 0.0
        self.persist = discovery.load_state()

        # Failover cascade bookkeeping.
        self._candidates: List[dict] = []   # last ranked list, nearest first
        self.failover_count = 0             # how many nodes were skipped to get here
        self.route_via = ""                 # "known" | "edge" | "broadcast" | "scan" | "manual"

        # Manual override.
        self.manual_locked = False
        self._manual_addr = ""

    # ---------------------------------------------------------- listeners
    def subscribe(self, fn):
        self._listeners.append(fn)

    def unsubscribe(self, fn):
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self):
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    @property
    def online(self) -> bool:
        return self.state == "online"

    @property
    def searching(self) -> bool:
        return self._searching

    @property
    def mode(self) -> str:
        return "manual" if self.manual_locked else "automatic"

    @property
    def safety_words(self) -> str:
        return self.client.safety_words or ""

    @property
    def known_coaches(self) -> dict:
        return self.persist.setdefault("coaches", {})

    @property
    def location_label(self) -> str:
        """What the network panel shows for "where am I connected" — a
        city or a routing label, never a raw IP/hostname. Manually typed
        addresses are the one exception: the person typed it themselves,
        so echoing it back is just confirmation, not a leak."""
        if not self.info:
            return ""
        if self.manual_locked:
            return f"Manual · {self.info.get('host', '')}"
        city = self.info.get("city")
        if city:
            return city
        if self.info.get("edge_name"):
            return self.info["edge_name"]
        if discovery.is_private_host(self.info.get("host", "")):
            return "Local network"
        return "Cloud coach"

    def forget_coach(self):
        """Deliberately re-pair with whatever is answering now."""
        host = (self.info or {}).get("host")
        if host:
            self.known_coaches.pop(host, None)
            discovery.save_state(self.persist)
        self.security_warning = ""
        self.discover()

    # ---------------------------------------------------------- discovery
    def start(self):
        self.discover()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def discover(self, manual_host: Optional[str] = None):
        """Run (or re-run) discovery. `manual_host` — a raw string typed by
        the person — forces manual mode for this and every future
        reconnect, until `clear_manual_override()` is called."""
        with self._lock:
            if self._searching:
                return
            self._searching = True
        threading.Thread(target=self._discover, args=(manual_host,), daemon=True).start()

    def connect_manual(self, address: str) -> bool:
        """Escape hatch: bypass auto-discovery and force a connection to a
        specific IP/host or ws(s)/http(s) URI. Returns False immediately if
        the address doesn't even parse; the actual connection attempt still
        happens on a background thread same as `discover()`."""
        candidate = discovery.parse_manual_address(address, default_port=config.SERVER_PORT)
        if not candidate:
            self.state = "offline"
            self.status_text = "That doesn't look like a valid address"
            self._notify()
            return False
        self._manual_addr = address.strip()
        self.discover(manual_host=address.strip())
        return True

    def clear_manual_override(self):
        """Go back to automatic nearest-node routing."""
        self.manual_locked = False
        self._manual_addr = ""
        self.discover()

    def _discover(self, manual_host: Optional[str]):
        try:
            self.state = "searching"
            self.status_text = ("Connecting to your server…" if manual_host
                                else "Looking for your coach…")
            self.security_warning = ""
            self._notify()

            def on_status(msg):
                self.status_text = msg
                self._notify()

            if manual_host is not None:
                self.manual_locked = True
                self._manual_addr = manual_host
                candidate = discovery.parse_manual_address(manual_host, default_port=config.SERVER_PORT)
                if not candidate:
                    self.state = "offline"
                    self.status_text = "That doesn't look like a valid address"
                    self._notify()
                    return
                info = discovery.probe(candidate["host"], candidate["port"], timeout=4.0,
                                       use_https=candidate.get("https"))
                if not info:
                    self.state = "offline"
                    self.status_text = "Couldn't reach that address"
                    self._notify()
                    return
                info["https"] = candidate.get("https", info.get("https", False))
                info["via"] = "manual"
                candidates = [info]
            elif self.manual_locked and self._manual_addr:
                # Heartbeat-triggered reconnect while locked to a manual
                # address: retry that one address, not the whole internet.
                candidate = discovery.parse_manual_address(self._manual_addr,
                                                            default_port=config.SERVER_PORT)
                info = candidate and discovery.probe(candidate["host"], candidate["port"],
                                                     timeout=4.0, use_https=candidate.get("https"))
                if not info:
                    self.state = "offline"
                    self.status_text = "Manual server unreachable — retrying…"
                    self._notify()
                    return
                info["https"] = candidate.get("https", info.get("https", False))
                info["via"] = "manual"
                candidates = [info]
            else:
                hosts = []
                if self.persist.get("last_host"):
                    hosts.append(self.persist["last_host"])
                hosts += config.PREFERRED_HOSTS

                candidates = discovery.find_server(extra_hosts=hosts, port=config.SERVER_PORT,
                                                   on_status=on_status, timeout=10.0,
                                                   edge_servers=config.EDGE_SERVERS,
                                                   return_all=True) or []

            self._candidates = list(candidates)
            if not candidates:
                self.state = "offline"
                self.status_text = "No coach found — still looking"
                self._notify()
                return

            self._run_cascade(candidates)
        finally:
            with self._lock:
                self._searching = False

    def _run_cascade(self, candidates: List[dict]):
        """Strict failover: try candidates nearest-first. A health check
        having already passed doesn't guarantee the handshake will — a
        node can go unhealthy in the gap, or simply time out — so any
        non-security failure moves to the next candidate immediately.
        An identity mismatch stops the whole cascade: that's not "this
        node is down", it's "something answering as a trusted coach isn't
        the coach we trusted", and no other node gets tried until a person
        looks at it."""
        for i, info in enumerate(candidates):
            self.failover_count = i
            if i > 0:
                self.status_text = (f"{info.get('edge_name') or info.get('city') or info['host']} "
                                    f"unavailable — trying next node…")
                self._notify()
            outcome = self._attempt(info)
            if outcome == "ok":
                return
            if outcome == "untrusted":
                return  # never fall through to another node on a trust failure
            # outcome == "failed": instant failover, keep going
        self.state = "offline"
        self.status_text = ("No healthy coach server reachable"
                            if len(candidates) > 1 else "No coach found — still looking")
        self._notify()

    def _attempt(self, info: dict) -> str:
        """One candidate, one handshake attempt. Returns "ok", "untrusted"
        or "failed" — the cascade above decides what to do with that."""
        host, port = info["host"], info.get("port", config.SERVER_PORT)
        use_https = info.get("https", False)
        if config.REQUIRE_HTTPS_FOR_PUBLIC_HOSTS and not discovery.is_private_host(host) and not use_https:
            # Discovery already refuses to probe a public host over plain
            # HTTP (see discovery.probe), so reaching this branch would mean
            # something upstream handed us a host it shouldn't have. Refuse
            # rather than silently connecting insecurely, and don't treat
            # this as a reason to try a different node — it's a policy
            # violation, not a dead server.
            self.state = "offline"
            self.status_text = "Refused an unencrypted connection to a public server"
            self._notify()
            return "untrusted"
        self.client.retarget(host, port, use_https=use_https)
        expected = self.known_coaches.get(host, {}).get("fingerprint", "")
        try:
            offer = self.client.connect(expect_fingerprint=expected)
        except SecurityError as e:
            # Do not fall back to an unverified connection. Ever.
            self.state = "untrusted"
            self.security_warning = str(e)
            self.status_text = "Coach identity doesn't match"
            self._notify()
            return "untrusted"
        except SpotterClientError:
            return "failed"

        first_time = not expected
        self.known_coaches[host] = {
            "fingerprint": self.client.fingerprint,
            "words": self.client.safety_words,
            "seen": time.time(),
        }
        self.persist["last_host"] = host
        discovery.save_state(self.persist)

        self.info = {**info, "version": offer.get("version", ""),
                     "catalog": offer.get("exercises", []),
                     "roadmap": offer.get("roadmap", []),
                     "first_time": first_time}
        self.route_via = info.get("via", "")
        self.rtt_ms = self.client.last_rtt_ms
        self.state = "online"
        self.status_text = ("Paired with a new coach" if first_time
                            else "Secure session ready")
        self.last_ok = time.time()
        self._notify()
        return "ok"

    # ---------------------------------------------------------- heartbeat
    def set_live_mode(self, live: bool):
        self._live_mode = live

    def frame_ok(self, rtt_ms: Optional[float]):
        self.last_ok = time.time()
        if rtt_ms is not None:
            self.rtt_ms = rtt_ms
        if self.state != "online":
            self.state = "online"
            self._notify()

    def frame_failed(self):
        if self.state == "online" and time.time() - self.last_ok > config.REDISCOVER_AFTER_S:
            self.state = "offline"
            self.status_text = "Connection lost — reconnecting"
            self._notify()
            self._quick_failover()

    def _quick_failover(self):
        """The current node just died mid-session. Before paying for a
        full fresh discovery pass (LAN broadcast + subnet sweep can take
        seconds), try the rest of the last ranked candidate list — it was
        current a few seconds ago and is usually still good. Falls back to
        a full `discover()` only if none of those work either, or if we're
        locked to a single manual address."""
        if self.manual_locked:
            self.discover()
            return
        current_host = (self.info or {}).get("host")
        rest = [c for c in self._candidates if c.get("host") != current_host]
        if not rest:
            self.discover()
            return
        with self._lock:
            if self._searching:
                return
            self._searching = True
        threading.Thread(target=self._cascade_thread, args=(rest,), daemon=True).start()

    def _cascade_thread(self, candidates: List[dict]):
        try:
            self.state = "searching"
            self.status_text = "Failing over to the next-nearest node…"
            self._notify()
            self._run_cascade(candidates)
            if self.state != "online":
                # The cached list is stale — do a real rediscovery.
                self._discover(None)
        finally:
            with self._lock:
                self._searching = False

    def _heartbeat(self):
        probe = SpotterClient(timeout=2.5)
        while not self._stop.is_set():
            time.sleep(3.0)
            if self._live_mode or self.state == "untrusted":
                continue
            if self.state == "online" and self.info:
                probe.retarget(self.info["host"], self.info.get("port", config.SERVER_PORT),
                               use_https=self.info.get("https", False))
                if probe.health():
                    self.rtt_ms = round(probe.last_rtt_ms or 0, 1)
                    self.last_ok = time.time()
                    self._notify()
                elif time.time() - self.last_ok > config.REDISCOVER_AFTER_S:
                    self.state = "offline"
                    self.status_text = "Connection lost — reconnecting"
                    self._notify()
                    self._quick_failover()
            elif self.state in ("offline", "idle"):
                self.discover()

    def ensure_session(self) -> bool:
        """Re-open a session if the server dropped ours mid-set."""
        if self.client.connected:
            return True
        if not self.info:
            return False
        host = self.info["host"]
        try:
            self.client.connect(
                expect_fingerprint=self.known_coaches.get(host, {}).get("fingerprint", ""))
            return True
        except SpotterClientError:
            return False

    def stop(self):
        self._stop.set()
