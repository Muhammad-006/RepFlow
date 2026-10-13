"""
discovery.py - finds the RepFlow AI Coach Server automatically, and picks
the fastest one when more than one is reachable.

Nobody types an IP address. When the app starts it races several
strategies and connects to whichever server actually answers fastest:

  1. Known addresses  - raced against each other by round-trip latency, not
                        "first to answer": the last server that worked
                        (cached on the phone), the laptop IP baked in at
                        APK build time (build_config.py), well-known LAN
                        addresses (127.0.0.1 for USB `adb reverse`,
                        10.0.2.2 for the Android emulator, 192.168.137.1
                        for Windows Mobile Hotspot), and any always-on
                        cloud servers in config.EDGE_SERVERS (e.g. one per
                        AWS region). Whichever of these answers
                        GET /v1/health first *wins the race*, which in
                        practice means whichever has the lowest latency
                        right now — this is RepFlow's "automatic routing to
                        the nearest edge node": no region is hand-picked,
                        anywhere. See EDGE_ROUTING.md for the full design.
  2. UDP broadcast    - asks "REPFLOW_DISCOVER?" on every local network; the
                        server replies (see server/discovery.py). Also
                        listens for the server's 1-second beacon. Only
                        reached if none of the known addresses answered.
  3. Subnet sweep     - probes port 8765 on every address of each local /24
                        network (covers phone-hotspot setups and networks
                        that block broadcast). Last resort.

Strict protocol note: a host that isn't on a private network (see
`is_private_host`) is refused unless it answers over HTTPS. This is
enforced here, at discovery time, and again in connection.py before a
session opens — a coach's identity still has to check out via the
existing X25519 handshake and safety-word pinning either way (HTTPS adds
transport security under the app's own end-to-end encryption, it doesn't
replace it).

Pure standard library -- safe to bundle into the Android APK.
"""

from __future__ import annotations

import concurrent.futures as cf
import http.client
import ipaddress
import json
import os
import socket
import ssl
import sys
import threading
import time
from typing import Callable, List, Optional

try:  # optional: used only to list network interfaces on Linux/Android
    import ctypes
except Exception:  # pragma: no cover
    ctypes = None

DEFAULT_PORT = 8765
DISCOVERY_PORT = 8766
BEACON_PORT = 8767
MAGIC = b"REPFLOW_DISCOVER?"

WELL_KNOWN_HOSTS = ["127.0.0.1", "10.0.2.2", "192.168.137.1"]


# ----------------------------------------------------------------------
# Strict protocol enforcement
# ----------------------------------------------------------------------
def is_private_host(host: str) -> bool:
    """Is this address on a network we trust to carry plain HTTP?

    Loopback (USB), link-local, and the private ranges LAN discovery and
    hotspots use — everything else (a cloud edge server's public IP or
    domain) must be reached over HTTPS. A hostname that isn't a literal IP
    is treated as public (it's never one of discovery's own LAN candidates,
    which are always numeric)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in ("localhost",)
    return ip.is_loopback or ip.is_link_local or ip.is_private


# ----------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------
def probe(host: str, port: int = DEFAULT_PORT, timeout: float = 1.0,
          use_https: Optional[bool] = None) -> Optional[dict]:
    """GET /v1/health. Returns {service,status,version,ready} plus
    host/port/rtt_ms/https, or None.

    Deliberately the *only* thing we ask for before the handshake: the
    server no longer publishes its hostname, OS or local addresses, so
    discovery cannot be used to fingerprint the machine it runs on.

    `use_https=None` means "decide from the address": private/LAN hosts
    default to plain HTTP (matches every coach server this app has ever
    shipped against), anything else defaults to HTTPS and — unless the
    caller explicitly passed `use_https=False` — a public host is refused
    outright rather than silently tried over plain HTTP. Encrypting the
    transport doesn't add security frames don't already have (they're
    sealed end-to-end at the application layer regardless), but a coach
    reachable over the open internet gets the extra layer as a strict
    baseline, not an opt-in.
    """
    private = is_private_host(host)
    if use_https is None:
        use_https = not private
    if use_https is False and not private:
        return None
    t0 = time.perf_counter()
    conn = None
    try:
        if use_https:
            conn = http.client.HTTPSConnection(host, port, timeout=timeout,
                                                context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/v1/health")
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            return None
        info = json.loads(data.decode("utf-8"))
        if info.get("service") != "repflow":
            return None
        info["host"] = host
        info["port"] = port
        info["https"] = use_https
        info["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return info
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _port_open(host: str, port: int, timeout: float) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


# ----------------------------------------------------------------------
# Local addresses (works on Android without extra packages)
# ----------------------------------------------------------------------
def _getifaddrs_ipv4() -> List[str]:
    """IPv4 addresses of all interfaces via libc getifaddrs (Linux/Android)."""
    if ctypes is None or not (sys.platform.startswith("linux") or hasattr(sys, "getandroidapilevel")):
        return []

    class sockaddr(ctypes.Structure):
        _fields_ = [("sa_family", ctypes.c_uint16), ("sa_data", ctypes.c_uint8 * 14)]

    class ifaddrs(ctypes.Structure):
        pass

    ifaddrs._fields_ = [
        ("ifa_next", ctypes.POINTER(ifaddrs)),
        ("ifa_name", ctypes.c_char_p),
        ("ifa_flags", ctypes.c_uint),
        ("ifa_addr", ctypes.POINTER(sockaddr)),
        ("ifa_netmask", ctypes.POINTER(sockaddr)),
        ("ifa_ifu", ctypes.POINTER(sockaddr)),
        ("ifa_data", ctypes.c_void_p),
    ]
    libc = None
    for name in ("libc.so", "libc.so.6"):
        try:
            libc = ctypes.CDLL(name, use_errno=True)
            break
        except OSError:
            continue
    if libc is None or not hasattr(libc, "getifaddrs"):
        return []
    head = ctypes.POINTER(ifaddrs)()
    if libc.getifaddrs(ctypes.byref(head)) != 0:
        return []
    ips = []
    try:
        p = head
        while p:
            ifa = p.contents
            if ifa.ifa_addr and ifa.ifa_addr.contents.sa_family == socket.AF_INET:
                raw = bytes(ifa.ifa_addr.contents.sa_data[2:6])
                ips.append(socket.inet_ntoa(raw))
            p = ifa.ifa_next
    finally:
        libc.freeifaddrs(head)
    return ips


def local_ipv4s() -> List[str]:
    ips = set()
    try:
        ips.update(_getifaddrs_ipv4())
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    for target in ("8.8.8.8", "192.168.1.1", "10.255.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 80))
            ips.add(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    out = []
    for ip in ips:
        if ip.startswith(("127.", "169.254.", "0.")):
            continue
        a, b = (int(x) for x in ip.split(".")[:2])
        private = a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)
        if private:          # mobile-data (carrier) addresses can't reach a laptop
            out.append(ip)
    return sorted(out)


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
def _udp_discover(stop: threading.Event, found: Callable[[str, int], None], duration: float = 3.0):
    ips = local_ipv4s()
    targets = {"255.255.255.255"} | {".".join(ip.split(".")[:3] + ["255"]) for ip in ips}

    def listen_beacon():
        try:
            b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            b.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            b.bind(("", BEACON_PORT))
            b.settimeout(0.3)
        except OSError:
            return
        end = time.time() + duration + 1.0
        while not stop.is_set() and time.time() < end:
            try:
                data, addr = b.recvfrom(2048)
                info = json.loads(data.decode("utf-8"))
                if info.get("service") == "repflow":
                    found(addr[0], int(info.get("port", DEFAULT_PORT)))
            except (socket.timeout, ValueError, OSError):
                continue
        b.close()

    threading.Thread(target=listen_beacon, daemon=True).start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.3)
    except OSError:
        return
    end = time.time() + duration
    next_send = 0.0
    while not stop.is_set() and time.time() < end:
        if time.time() >= next_send:
            for t in targets:
                try:
                    s.sendto(MAGIC, (t, DISCOVERY_PORT))
                except OSError:
                    pass
            next_send = time.time() + 0.8
        try:
            data, addr = s.recvfrom(2048)
            info = json.loads(data.decode("utf-8"))
            if info.get("service") == "repflow":
                found(addr[0], int(info.get("port", DEFAULT_PORT)))
        except (socket.timeout, ValueError, OSError):
            continue
    s.close()


def _sweep(stop: threading.Event, found: Callable[[str, int], None], port: int):
    ips = local_ipv4s()
    hosts = []
    for ip in ips:
        base = ".".join(ip.split(".")[:3])
        mine = int(ip.split(".")[3])
        # gateway-ish and nearby addresses first, then the rest
        order = [1, 2, 100, 101, 137] + [((mine + d) % 254) + 1 for d in range(254)]
        seen = set()
        for last in order:
            if 1 <= last <= 254 and last not in seen and f"{base}.{last}" != ip:
                seen.add(last)
                hosts.append(f"{base}.{last}")
    if not hosts:
        return

    def check(h):
        if stop.is_set():
            return
        if _port_open(h, port, 0.45):
            found(h, port)

    with cf.ThreadPoolExecutor(max_workers=48) as ex:
        for h in hosts:
            if stop.is_set():
                break
            ex.submit(check, h)


def race_and_rank(
    candidates: List[dict],
    window: float = 1.6,
    per_probe_timeout: float = 1.5,
) -> List[dict]:
    """Probe every candidate at once and return ALL healthy ones, sorted
    lowest-latency first.

    `candidates` is a list of `{"host", "port", "https"?, "name"?, "city"?}`
    dicts — LAN addresses, the last host that worked, and any configured
    `EDGE_SERVERS`. Unlike racing to see who answers *first* (subject to
    thread-scheduling noise for near-simultaneous replies), this waits out
    the window and then explicitly ranks by `rtt_ms`, so "nearest server"
    means what it says. This is also what makes a strict failover cascade
    possible: the caller gets the *whole* ordered list, not just the
    winner, so if the nearest node's connection drops after the health
    check passed, the next-best one is already known and ready — no need
    to re-run discovery from scratch.
    """
    results: List[dict] = []
    lock = threading.Lock()

    def probe_one(c: dict):
        info = probe(c["host"], c.get("port", DEFAULT_PORT), timeout=per_probe_timeout,
                     use_https=c.get("https"))
        if info:
            info["edge_name"] = c.get("name", "")
            info["city"] = c.get("city", "")
            with lock:
                results.append(info)

    threads = [threading.Thread(target=probe_one, args=(c,), daemon=True) for c in candidates]
    for t in threads:
        t.start()
    for t in threads:
        t.join(window)
    with lock:
        return sorted(results, key=lambda r: r.get("rtt_ms", float("inf")))


def race_known_hosts(
    candidates: List[dict],
    window: float = 1.6,
    per_probe_timeout: float = 1.5,
) -> Optional[dict]:
    """Back-compat wrapper: just the single fastest healthy candidate.

    Prefer `race_and_rank` for anything that wants failover — it returns
    every healthy candidate, not only the winner.
    """
    ranked = race_and_rank(candidates, window=window, per_probe_timeout=per_probe_timeout)
    return ranked[0] if ranked else None


# ----------------------------------------------------------------------
def find_server(
    extra_hosts: Optional[List[str]] = None,
    port: int = DEFAULT_PORT,
    timeout: float = 12.0,
    on_status: Optional[Callable[[str], None]] = None,
    edge_servers: Optional[List[dict]] = None,
    return_all: bool = False,
):
    """Returns the fastest-reachable server's /health info (with
    host/port/rtt_ms/https/city, and "via" = how it was found), or None
    after `timeout` seconds.

    `return_all=True` returns every healthy candidate instead, sorted
    lowest-latency first — this is what makes a strict failover cascade
    possible: the nearest node is tried first, and if it turns out to be
    unhealthy or the handshake fails, the caller already has the next-best
    node ranked and ready rather than needing a fresh discovery pass.
    """
    stop = threading.Event()
    result: dict = {}

    def status(msg):
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    known = []
    seen = set()
    for h in (extra_hosts or []) + WELL_KNOWN_HOSTS:
        key = (h, port)
        if h and key not in seen:
            seen.add(key)
            known.append({"host": h, "port": port})
    for e in (edge_servers or []):
        key = (e.get("host"), e.get("port", port))
        if e.get("host") and key not in seen:
            seen.add(key)
            known.append(dict(e))

    status("Checking known addresses…")
    ranked = race_and_rank(known)
    if ranked:
        for r in ranked:
            r["via"] = "edge" if r.get("edge_name") else "known"
        return ranked if return_all else ranked[0]

    # Nothing known answered — fall back to LAN discovery, which realistically
    # only ever turns up one server, so first-to-answer is the whole race.
    lock = threading.Lock()
    tried = set()

    def found(host: str, p: int, via: str = ""):
        key = (host, p)
        with lock:
            if stop.is_set() or key in tried:
                return
            tried.add(key)
        info = probe(host, p, timeout=1.5)
        if info:
            with lock:
                if not result:
                    info["via"] = via
                    result.update(info)
                    stop.set()

    status("Broadcasting on local network…")
    udp = threading.Thread(
        target=_udp_discover, args=(stop, lambda h, p: found(h, p, "broadcast"), 2.5), daemon=True
    )
    udp.start()
    udp.join(1.2)
    if not stop.is_set():
        status("Scanning network for the AI server…")
        sweep = threading.Thread(target=_sweep, args=(stop, lambda h, p: found(h, p, "scan"), port), daemon=True)
        sweep.start()

    end = time.time() + timeout
    while not stop.is_set() and time.time() < end:
        time.sleep(0.05)
    stop.set()
    if return_all:
        return [dict(result)] if result else []
    return dict(result) if result else None


# ----------------------------------------------------------------------
# Manual override — an explicit escape hatch from auto-discovery
# ----------------------------------------------------------------------
def parse_manual_address(raw: str, default_port: int = DEFAULT_PORT) -> Optional[dict]:
    """Turn whatever a person typed into the manual-connection box into a
    `{"host", "port", "https"}` candidate, or None if it's not parseable.

    Accepts: a bare IP or hostname ("192.168.1.20"), host:port
    ("192.168.1.20:8765"), or a full URI ("ws://", "wss://", "http://",
    "https://" + host[:port]). A scheme is the only way to force plain
    HTTP to a non-private host — the strict-protocol rule in
    `is_private_host`/`probe` still applies after parsing, this function
    only extracts host/port/scheme.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    https: Optional[bool] = None
    rest = raw
    for scheme, val in (("wss://", True), ("https://", True), ("ws://", False), ("http://", False)):
        if raw.lower().startswith(scheme):
            https = val
            rest = raw[len(scheme):]
            break
    rest = rest.split("/", 1)[0]  # drop any path
    if not rest:
        return None

    if ":" in rest:
        host, _, port_s = rest.rpartition(":")
        if not host or not port_s.isdigit():
            return None
        port = int(port_s)
    else:
        host = rest
        # No port given: an explicit https/wss scheme implies the standard
        # HTTPS port, everything else falls back to the coach's usual port.
        port = 443 if https else default_port

    if not host:
        return None
    # A loose sanity check — not a full RFC validation, just enough to
    # reject empty/garbage input before we ever open a socket.
    if not all(c.isalnum() or c in ".-_:" for c in host):
        return None

    if https is None:
        https = not is_private_host(host)
    return {"host": host, "port": port, "https": https}


# ----------------------------------------------------------------------
# Tiny persistent cache (last good server) -- works on Android & desktop
# ----------------------------------------------------------------------
def _cache_path() -> str:
    base = os.environ.get("FLET_APP_STORAGE_DATA") or os.path.join(os.path.expanduser("~"), ".repflow")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        pass
    return os.path.join(base, "repflow_state.json")


def load_state() -> dict:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict):
    try:
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


if __name__ == "__main__":  # quick manual test:  python discovery.py
    print("local IPs:", local_ipv4s())
    t = time.time()
    info = find_server(on_status=print)
    print(f"{time.time() - t:.2f}s ->", info)
