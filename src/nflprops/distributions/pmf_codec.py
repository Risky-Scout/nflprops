"""BLOCK 2A: lossless, versioned, deterministic binary codec for the sparse
positive-mass side of an exact discrete PMF (`nflprops.distributions.pmf.RawPMF`).

This is a persistence-optimization codec only -- it never touches model
science. It encodes exactly the same ``(outcome, probability)`` pairs
`nflprops.orchestration.distribution_store` already persists one row per
outcome (`player_prop_distribution_outcomes`), as a single compact binary
blob instead, so a canonical distribution can be stored as ONE
`player_prop_distributions` row rather than ``outcome_count`` child rows.

Format (all integers/floats big-endian / network byte order, via `struct`,
so the byte representation is platform-independent and deterministic):

    header (10 bytes):
        magic          4s   b"NFPM"
        codec_version  H    uint16, currently 1
        outcome_count  I    uint32

    body (16 bytes * outcome_count, repeated in ascending outcome order):
        outcome        q    int64, signed
        probability    d    float64 -- packed/unpacked via `struct`, so a
                             decoded probability is bit-identical to the
                             float64 that was encoded (no string/JSON
                             round trip, no rounding, no quantization)

Sparse by construction: only strictly-positive-probability outcomes are
ever encoded (interior/exterior zero-probability outcomes are omitted, not
stored as zero rows) -- callers already only pass rows that satisfy this
(`nflprops.distributions.pmf.RawPMF` / `player_prop_distribution_outcomes.
p_raw > 0`), and `_validate_pmf_sequence` re-enforces it (a non-positive
probability fails closed on both encode and decode).

`encode_pmf`/`decode_pmf` both run the identical validation gate
(`_validate_pmf_sequence`): outcomes strictly increasing (so duplicate or
unsorted outcomes are rejected on either side), every probability finite
and strictly positive, and the probabilities summing to ``1.0`` within
`nflprops.distributions.pmf.NORMALIZATION_TOLERANCE`. A malformed payload
(bad magic, unsupported version, truncated/overlong body, or a validation
failure after decode) always raises `PMFCodecError` -- never silently
repaired, renormalized, truncated, or binned.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from nflprops.distributions.pmf import NORMALIZATION_TOLERANCE

#: Codec format magic. Never reused for a future, incompatible payload
#: shape -- a format change bumps `CODEC_VERSION` instead.
MAGIC: bytes = b"NFPM"

#: Current codec version. Embedded in every payload's header; `decode_pmf`
#: fails closed on any other value rather than guess at forward/backward
#: compatibility.
CODEC_VERSION: int = 1

_HEADER = struct.Struct(">4sHI")  # magic, codec_version, outcome_count
_RECORD = struct.Struct(">qd")  # outcome (int64), probability (float64)


class PMFCodecError(ValueError):
    """Encoding or decoding a compact sparse PMF payload failed: a bad
    magic/version, a truncated or overlong body, non-increasing or
    duplicate outcomes, a non-finite/non-positive probability, or a
    probability sum outside `nflprops.distributions.pmf.
    NORMALIZATION_TOLERANCE` of ``1.0``. Fails closed -- never repairs,
    renormalizes, truncates, or bins."""


@dataclass(frozen=True)
class DecodedPMF:
    """The exact ordered ``(outcome, probability)`` pairs recovered from a
    compact payload -- ascending by outcome, identical in content and
    order to what was encoded."""

    outcomes: tuple[int, ...]
    probabilities: tuple[float, ...]


def _validate_pmf_sequence(
    outcomes: tuple[int, ...], probabilities: tuple[float, ...]
) -> None:
    if len(outcomes) != len(probabilities):
        raise PMFCodecError(
            f"outcomes ({len(outcomes)}) and probabilities "
            f"({len(probabilities)}) must be the same length"
        )
    if len(outcomes) == 0:
        raise PMFCodecError(
            "a PMF must have at least one positive-probability outcome"
        )
    for i in range(1, len(outcomes)):
        if outcomes[i] <= outcomes[i - 1]:
            raise PMFCodecError(
                f"outcomes must be strictly increasing (sorted, no duplicates); "
                f"outcome[{i}]={outcomes[i]} <= outcome[{i - 1}]={outcomes[i - 1]}"
            )
    total = 0.0
    for p in probabilities:
        if not math.isfinite(p):
            raise PMFCodecError(f"probability {p!r} is not finite")
        if p <= 0.0:
            raise PMFCodecError(
                f"probability {p!r} is not strictly positive -- only "
                f"positive-mass outcomes are ever encoded"
            )
        total += p
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise PMFCodecError(
            f"probabilities sum to {total!r}, not 1.0 within "
            f"{NORMALIZATION_TOLERANCE}"
        )


def encode_pmf(outcomes: Sequence[int], probabilities: Sequence[float]) -> bytes:
    """Encode an ordered, strictly-increasing, positive-mass, normalized
    PMF into the compact binary payload. Raises `PMFCodecError` (nothing
    is returned) for duplicate/unsorted outcomes, a non-finite/
    non-positive probability, or a bad normalization sum -- the same gate
    `decode_pmf` re-applies on the way back out."""
    outcomes_t = tuple(int(o) for o in outcomes)
    probabilities_t = tuple(float(p) for p in probabilities)
    _validate_pmf_sequence(outcomes_t, probabilities_t)

    body = bytearray(_RECORD.size * len(outcomes_t))
    offset = 0
    for outcome, probability in zip(outcomes_t, probabilities_t, strict=True):
        _RECORD.pack_into(body, offset, outcome, probability)
        offset += _RECORD.size
    header = _HEADER.pack(MAGIC, CODEC_VERSION, len(outcomes_t))
    return header + bytes(body)


def decode_pmf(payload: bytes) -> DecodedPMF:
    """Decode a compact binary payload back into the exact ordered
    ``(outcome, probability)`` pairs that were encoded. Fails closed
    (`PMFCodecError`) on a bad magic/version, a length that disagrees with
    the header-declared outcome count, or any post-decode validation
    failure -- never silently accepts a malformed payload."""
    if len(payload) < _HEADER.size:
        raise PMFCodecError(
            f"payload too short for header: {len(payload)} bytes < "
            f"{_HEADER.size} bytes"
        )
    magic, version, count = _HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise PMFCodecError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != CODEC_VERSION:
        raise PMFCodecError(
            f"unsupported pmf_codec_version={version}; this build only "
            f"supports version {CODEC_VERSION}"
        )
    expected_len = _HEADER.size + count * _RECORD.size
    if len(payload) != expected_len:
        raise PMFCodecError(
            f"payload length {len(payload)} does not match header-declared "
            f"length {expected_len} for outcome_count={count}"
        )

    outcomes: list[int] = []
    probabilities: list[float] = []
    offset = _HEADER.size
    for _ in range(count):
        outcome, probability = _RECORD.unpack_from(payload, offset)
        outcomes.append(outcome)
        probabilities.append(probability)
        offset += _RECORD.size

    outcomes_t = tuple(outcomes)
    probabilities_t = tuple(probabilities)
    _validate_pmf_sequence(outcomes_t, probabilities_t)
    return DecodedPMF(outcomes=outcomes_t, probabilities=probabilities_t)


def payload_sha256(payload: bytes) -> str:
    """Deterministic SHA-256 over the exact encoded bytes."""
    return hashlib.sha256(payload).hexdigest()
