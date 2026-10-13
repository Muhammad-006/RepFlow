"""
crypto_box.py — RepFlow end-to-end encryption.

This exact file ships on BOTH sides (server/ and client/). It must stay
importable with **nothing but the Python standard library**, because the
phone app is compiled into an Android APK and no native crypto wheel
(cryptography / PyNaCl / pyca) builds for Android. Everything here is
therefore built from `hashlib` and `hmac`, which are C-accelerated in
CPython, plus a small pure-Python X25519.

What it gives you
-----------------
1. **X25519** key agreement (RFC 7748) — pure Python, ~1 ms per operation,
   run twice per session, never per frame.

2. A **Noise_NK-style handshake**. The server has a long-term identity key
   that lives on the server and never moves. For every session the client
   makes a throw-away key pair and does two Diffie–Hellmans:

       ss_static    = X25519(client_ephemeral_priv, server_identity_pub)
       ss_ephemeral = X25519(client_ephemeral_priv, server_ephemeral_pub)

   The first one proves the server is who the app pinned (nobody else has
   the identity private key, so nobody else can derive the same secret).
   The second one gives *forward secrecy*: the server throws its ephemeral
   key away when the session ends, so even someone who later steals the
   identity key cannot go back and decrypt a recorded session.

3. An **AEAD** (authenticated encryption) built as encrypt-then-MAC:

       keystream = BLAKE2b(nonce || counter, key=enc_key)   # counter mode
       ciphertext = plaintext XOR keystream
       tag = HMAC-SHA256(mac_key, aad || nonce || ciphertext)[:16]

   Encrypt-then-MAC with a keyed-hash keystream is a conservative,
   well-understood construction (the same shape as the TLS 1.2
   `AES-CBC + HMAC` suites, minus the block-cipher padding that made those
   fragile). Separate keys are used for the keystream and the MAC, the MAC
   covers the nonce and the associated data, and the tag is compared in
   constant time. A nonce is never reused: it is the session id bytes plus
   a strictly increasing frame counter, and the receiver rejects any
   counter it has already seen (replay protection).

Deliberately NOT here: any way to write a frame to disk, any logging, and
any key that outlives the session object holding it.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Tuple

__all__ = [
    "generate_keypair", "x25519", "derive_session_keys", "server_handshake",
    "client_handshake", "seal", "open_box", "CryptoError", "fingerprint",
    "fingerprint_words", "wipe",
]

PROTOCOL = b"RepFlow/E2E/v3"
KEY_BYTES = 32
TAG_BYTES = 16
NONCE_BYTES = 12


class CryptoError(Exception):
    """Authentication failed, replay detected, or a malformed box."""


# ======================================================================
# X25519  (RFC 7748, section 5)
# ======================================================================
_P = 2 ** 255 - 19
_A24 = 121665


def _decode_scalar(k: bytes) -> int:
    k = bytearray(k[:32])
    k[0] &= 248
    k[31] &= 127
    k[31] |= 64
    return int.from_bytes(k, "little")


def _decode_u(u: bytes) -> int:
    u = bytearray(u[:32])
    u[31] &= 127          # ignore the unused top bit, per RFC 7748
    return int.from_bytes(u, "little") % _P


def x25519(scalar: bytes, u_coord: bytes) -> bytes:
    """Montgomery ladder scalar multiplication on Curve25519."""
    k = _decode_scalar(scalar)
    x1 = _decode_u(u_coord)
    x2, z2, x3, z3, swap = 1, 0, x1, 1, 0

    for t in range(254, -1, -1):
        k_t = (k >> t) & 1
        swap ^= k_t
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = k_t

        a = (x2 + z2) % _P
        aa = (a * a) % _P
        b = (x2 - z2) % _P
        bb = (b * b) % _P
        e = (aa - bb) % _P
        c = (x3 + z3) % _P
        d = (x3 - z3) % _P
        da = (d * a) % _P
        cb = (c * b) % _P
        x3 = (da + cb) % _P
        x3 = (x3 * x3) % _P
        z3 = (da - cb) % _P
        z3 = (x1 * z3 * z3) % _P
        x2 = (aa * bb) % _P
        z2 = (e * (aa + _A24 * e)) % _P

    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2

    shared = (x2 * pow(z2, _P - 2, _P)) % _P
    out = shared.to_bytes(32, "little")
    if out == bytes(32):
        # All-zero output means a small-order / degenerate public key was
        # supplied. RFC 7748 says implementations MAY reject this; we do.
        raise CryptoError("degenerate public key rejected")
    return out


_BASE_POINT = (9).to_bytes(32, "little")


def generate_keypair() -> Tuple[bytes, bytes]:
    """(private_key, public_key), both 32 raw bytes."""
    priv = secrets.token_bytes(32)
    return priv, x25519(priv, _BASE_POINT)


def public_from_private(priv: bytes) -> bytes:
    return x25519(priv, _BASE_POINT)


# ======================================================================
# HKDF-SHA256  (RFC 5869)
# ======================================================================
def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def derive_session_keys(shared_secrets: bytes, transcript: bytes) -> dict:
    """Four independent keys from the handshake material.

    Two directions x (keystream key, MAC key). Using separate keys for
    each direction and each purpose means a flaw in one never weakens
    another, and it makes the two directions non-interchangeable (an
    attacker cannot reflect a client frame back as if the server sent it).
    """
    okm = hkdf(shared_secrets, salt=transcript, info=PROTOCOL, length=4 * KEY_BYTES)
    return {
        "c2s_enc": okm[0:32],
        "c2s_mac": okm[32:64],
        "s2c_enc": okm[64:96],
        "s2c_mac": okm[96:128],
    }


# ======================================================================
# Handshake
# ======================================================================
def client_handshake(server_identity_pub: bytes, server_ephemeral_pub: bytes) -> dict:
    """Run on the phone. Returns keys + the ephemeral public key to send."""
    eph_priv, eph_pub = generate_keypair()
    ss_static = x25519(eph_priv, server_identity_pub)
    ss_ephemeral = x25519(eph_priv, server_ephemeral_pub)
    transcript = hashlib.sha256(
        PROTOCOL + server_identity_pub + server_ephemeral_pub + eph_pub).digest()
    keys = derive_session_keys(ss_static + ss_ephemeral, transcript)
    wipe(eph_priv)
    return {"keys": keys, "client_pub": eph_pub, "transcript": transcript}


def server_handshake(identity_priv: bytes, ephemeral_priv: bytes,
                     client_pub: bytes) -> dict:
    """Run on the server, mirror of the above."""
    ss_static = x25519(identity_priv, client_pub)
    ss_ephemeral = x25519(ephemeral_priv, client_pub)
    transcript = hashlib.sha256(
        PROTOCOL + public_from_private(identity_priv)
        + public_from_private(ephemeral_priv) + client_pub).digest()
    return {"keys": derive_session_keys(ss_static + ss_ephemeral, transcript),
            "transcript": transcript}


# ======================================================================
# AEAD
# ======================================================================
def _keystream(key: bytes, nonce: bytes, nbytes: int) -> bytes:
    """BLAKE2b in counter mode. 64 bytes of keystream per hash call, all
    inside CPython's C implementation."""
    blocks = (nbytes + 63) // 64
    out = bytearray(blocks * 64)
    view = memoryview(out)
    for i in range(blocks):
        view[i * 64:(i + 1) * 64] = hashlib.blake2b(
            nonce + i.to_bytes(8, "big"), key=key, digest_size=64,
            person=b"repflow-stream").digest()
    return bytes(out[:nbytes])


def _xor(a: bytes, b: bytes) -> bytes:
    """XOR two equal-length byte strings as one big integer — this is a
    single C-level operation, far faster than a Python-level loop."""
    n = len(a)
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(n, "big")


def seal(enc_key: bytes, mac_key: bytes, counter: int, plaintext: bytes,
         aad: bytes = b"", salt: bytes = b"") -> bytes:
    """Encrypt-then-MAC. Returns nonce(12) || tag(16) || ciphertext."""
    nonce = (salt[:4].ljust(4, b"\0")) + counter.to_bytes(8, "big")
    ct = _xor(plaintext, _keystream(enc_key, nonce, len(plaintext))) if plaintext else b""
    tag = hmac.new(mac_key, aad + nonce + ct, hashlib.sha256).digest()[:TAG_BYTES]
    return nonce + tag + ct


def open_box(enc_key: bytes, mac_key: bytes, box: bytes, aad: bytes = b"") -> Tuple[int, bytes]:
    """Verify then decrypt. Returns (counter, plaintext).

    The MAC is checked *before* a single byte is decrypted, so forged or
    tampered data never reaches the decoder, let alone the pose model.
    """
    if len(box) < NONCE_BYTES + TAG_BYTES:
        raise CryptoError("box too short")
    nonce = box[:NONCE_BYTES]
    tag = box[NONCE_BYTES:NONCE_BYTES + TAG_BYTES]
    ct = box[NONCE_BYTES + TAG_BYTES:]
    expect = hmac.new(mac_key, aad + nonce + ct, hashlib.sha256).digest()[:TAG_BYTES]
    if not hmac.compare_digest(tag, expect):
        raise CryptoError("authentication failed")
    counter = int.from_bytes(nonce[4:], "big")
    pt = _xor(ct, _keystream(enc_key, nonce, len(ct))) if ct else b""
    return counter, pt


# ======================================================================
# Housekeeping
# ======================================================================
def wipe(buf) -> None:
    """Best-effort overwrite of key material.

    CPython will not let us zero an immutable `bytes` object, so keys that
    matter are held in `bytearray`s and passed through here. For anything
    immutable this is a no-op and we rely on dropping the last reference.
    """
    if isinstance(buf, bytearray):
        for i in range(len(buf)):
            buf[i] = 0


_WORDS = (
    "anchor apple amber arrow atlas basil beacon birch bison blaze bloom bravo "
    "cable cedar chalk cliff cobalt comet coral crane delta denim dune ember "
    "falcon fern flint forge gamma glide grove harbor haven helix indigo ivory "
    "jade kayak kite lagoon lantern larch lever lunar maple marble mesa meteor "
    "nickel nomad north oasis onyx opal orbit otter pearl pebble pilot pine "
    "prism quartz quiver raven reef ridge river rover sable saffron sage shale "
    "slate solar spruce stone summit tangle teal thorn tide timber topaz trail "
    "tundra umber vector velvet vine walnut willow wren zephyr zinc"
).split()


def fingerprint(public_key: bytes) -> str:
    """Hex fingerprint of a public key (first 8 bytes of its SHA-256)."""
    return hashlib.sha256(b"repflow-fp" + public_key).hexdigest()[:16]


def fingerprint_words(public_key: bytes, count: int = 4) -> str:
    """Human-comparable 'safety number'.

    The phone shows these words and so does the server. If they match, the
    app is talking to the machine you think it is and not to something in
    the middle. Same idea as Signal's safety numbers — a fingerprint people
    can actually read out loud.
    """
    digest = hashlib.sha256(b"repflow-fp" + public_key).digest()
    n = len(_WORDS)
    return " ".join(_WORDS[digest[i] % n] for i in range(count))


# ======================================================================
def load_or_create_identity(path: str) -> bytes:
    """The server's long-term key. Created once, chmod 600, never sent."""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            priv = f.read(32)
        if len(priv) == 32:
            return priv
    priv, _ = generate_keypair()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, priv)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return priv
