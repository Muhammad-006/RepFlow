"""
discovery.py - Zero-config "find my laptop" for the phone app.

The phone never needs an IP address typed in. Two mechanisms run together:

  * Responder  - listens on UDP DISCOVERY_PORT. The app broadcasts
                 b"REPFLOW_DISCOVER" and we reply (unicast) with our HTTP
                 port + info; the app takes our IP from the reply's source.
  * Beacon     - every second we broadcast the same info to
                 BEACON_PORT on every local network (same Wi-Fi, laptop
                 hotspot, or phone hotspot), for apps listening passively.

Stdlib only.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Callable, List

log = logging.getLogger("repflow.discovery")

DISCOVERY_PORT = 8766
BEACON_PORT = 8767
MAGIC = b"REPFLOW_DISCOVER"


def local_ipv4s() -> List[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    for probe in ("8.8.8.8", "192.168.1.1", "10.255.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 80))
            ips.add(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254."))


def _is_private(ip: str) -> bool:
    a, b = (int(x) for x in ip.split(".")[:2])
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def preferred_ips() -> List[str]:
    """LAN addresses first (what the phone will reach), others after."""
    ips = local_ipv4s()
    return sorted(ips, key=lambda ip: (not _is_private(ip), ip))


class DiscoveryService:
    def __init__(self, http_port: int, info_fn: Callable[[], dict]):
        self.http_port = http_port
        self.info_fn = info_fn
        self._stop = threading.Event()

    def _payload(self) -> bytes:
        info = {"service": "repflow", "port": self.http_port}
        try:
            info.update(self.info_fn())
        except Exception:
            pass
        return json.dumps(info).encode("utf-8")

    def start(self):
        threading.Thread(target=self._responder, name="discovery-responder", daemon=True).start()
        threading.Thread(target=self._beacon, name="discovery-beacon", daemon=True).start()

    def stop(self):
        self._stop.set()

    def _responder(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
        except OSError as e:
            log.warning("Discovery responder disabled (port %d busy?): %s", DISCOVERY_PORT, e)
            return
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                continue
            if data.startswith(MAGIC):
                try:
                    sock.sendto(self._payload(), addr)
                    log.info("Discovery: answered %s", addr[0])
                except OSError:
                    pass

    def _beacon(self):
        while not self._stop.is_set():
            payload = self._payload()
            targets = {"255.255.255.255"}
            ips = local_ipv4s()
            for ip in ips:
                parts = ip.split(".")
                targets.add(".".join(parts[:3] + ["255"]))
            # One socket per local interface so the broadcast leaves on every
            # network (Windows only uses the primary adapter for 255.255.255.255).
            for src in ips + [""]:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    if src:
                        s.bind((src, 0))
                    for t in targets:
                        try:
                            s.sendto(payload, (t, BEACON_PORT))
                        except OSError:
                            pass
                    s.close()
                except OSError:
                    pass
            self._stop.wait(1.0)
