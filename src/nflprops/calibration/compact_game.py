"""Compact per-game calibration evidence for streaming Phase 10C3A
execution (execution/reporting plumbing only -- no model science).

A `LabeledGame` carries the full `GameSimulationResult` (every simulated
player x every draw x ~60 stat columns, ~1.7 GB per game at 20,000
draws). Everything the joint-game calibration pipeline ever reads from
that simulation is far smaller:

* the `(n_draws, len(FEATURE_NAMES))` joint-game feature matrix
  (`compute_draw_features`), and
* for every labeled (player, prop), the canonical per-draw outcome vector
  (`canonical_outcome_values`) -- stored losslessly as its sorted unique
  support plus one integer code per draw (`np.unique(..., return_inverse=True)`,
  exactly the factorization `build_weighted_pmf` performs), and
* the first-TD scorer per draw (for the real-game coherence check).

`CompactGame` holds exactly that, so the full draw table can be released
right after one game is replayed. Weighted PMFs rebuilt from it are
bit-identical to `build_weighted_pmf` on the full simulation: same support,
same per-draw bucket, same sequential in-order float64 accumulation
(`np.bincount` with weights sums in input order, as `np.add.at` does).
`tests/calibration/test_compact_game.py` locks that equivalence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nflprops.calibration.joint_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_NAMES,
    compute_draw_features,
)
from nflprops.calibration.weighted_pmf import (
    WeightedPMF,
    WeightedPMFError,
    validate_draw_weights,
)
from nflprops.distributions.pmf import NORMALIZATION_TOLERANCE, canonical_outcome_values
from nflprops.domain.enums import PropType
from nflprops.simulation.game import GameSimulationResult

if TYPE_CHECKING:
    from nflprops.calibration.challenger import LabeledGame, PropLabel

COMPACT_GAME_VERSION = "compact_game/v1"

_META = "meta.json"
_FEATURES = "features.npy"
_CODES = "codes.npy"
_FIRST_TD = "first_td_codes.npy"


class CompactGameError(ValueError):
    """A compact game failed a structural or integrity check."""


@dataclass(frozen=True)
class CompactGame:
    """Duck-type of `LabeledGame` for every calibration consumer: same
    identity/chronology/label fields, with the simulation replaced by the
    sufficient per-draw arrays described in the module docstring."""

    game_id: str
    as_of: datetime
    outcome_available_at: datetime
    injury_data_available: bool
    labels: tuple[PropLabel, ...]
    n_draws: int
    model_version: str
    #: `(n_draws, len(FEATURE_NAMES))` float64, `compute_draw_features` output.
    features: np.ndarray
    #: Per label (same order as `labels`): ascending unique outcome values.
    supports: tuple[np.ndarray, ...]
    #: `(len(labels), n_draws)` unsigned codes into `supports[i]`.
    codes: np.ndarray
    #: Per label: the simulated player's position group.
    position_groups: tuple[str, ...]
    first_td_labels: tuple[str, ...]
    first_td_codes: np.ndarray

    def draw_features(self) -> np.ndarray:
        return self.features

    def label_codes(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.supports[index], self.codes[index]


def _code_dtype(max_support: int) -> type[np.unsignedinteger[Any]]:
    if max_support <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def compact_from_labeled_game(game: LabeledGame) -> CompactGame:
    """Extract the sufficient arrays from one full `LabeledGame`. Every
    outcome vector comes from the certified `canonical_outcome_values`
    (per-player sub-results only narrow the rows `prop_values` filters)."""
    result = game.simulation
    by_player = result.player_draws.partition_by("player_id", as_dict=True)
    sub_results: dict[str, GameSimulationResult] = {}
    positions: dict[str, str] = {}

    def player_result(player_id: str) -> GameSimulationResult:
        if player_id not in sub_results:
            frame = by_player.get((player_id,))
            if frame is None:
                raise CompactGameError(f"{game.game_id}: player {player_id!r} not simulated")
            sub_results[player_id] = GameSimulationResult(
                game_id=result.game_id,
                model_version=result.model_version,
                as_of=result.as_of,
                n_draws=result.n_draws,
                player_draws=frame,
                team_draws=result.team_draws,
                first_td_player=result.first_td_player,
            )
            groups = frame["position_group"].unique().to_list()
            if len(groups) != 1:
                raise CompactGameError(f"{game.game_id}: player {player_id!r} position {groups}")
            positions[player_id] = str(groups[0])
        return sub_results[player_id]

    supports: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    for label in game.labels:
        values = canonical_outcome_values(
            player_result(label.player_id), label.player_id, label.prop_type
        )
        outcomes, inverse = np.unique(values, return_inverse=True)
        supports.append(outcomes.astype(np.int64))
        inverses.append(inverse)
    dtype = _code_dtype(max(len(s) for s in supports))
    codes = np.vstack([inv.astype(dtype) for inv in inverses])

    td_labels, td_inverse = np.unique(np.asarray(result.first_td_player, dtype=object),
                                      return_inverse=True)
    return CompactGame(
        game_id=game.game_id,
        as_of=game.as_of,
        outcome_available_at=game.outcome_available_at,
        injury_data_available=game.injury_data_available,
        labels=game.labels,
        n_draws=result.n_draws,
        model_version=result.model_version,
        features=compute_draw_features(result),
        supports=tuple(supports),
        codes=codes,
        position_groups=tuple(positions[label.player_id] for label in game.labels),
        first_td_labels=tuple(str(x) for x in td_labels.tolist()),
        first_td_codes=td_inverse.astype(_code_dtype(len(td_labels))),
    )


def compact_weighted_pmf(game: CompactGame, index: int, weights: np.ndarray) -> WeightedPMF:
    """`build_weighted_pmf` for label `index`, from the compact arrays --
    same validation, same support, bit-identical probabilities."""
    label = game.labels[index]
    prop = PropType(label.prop_type)
    outcomes_arr, codes = game.label_codes(index)
    w = validate_draw_weights(weights, game.n_draws)
    probs_arr = np.bincount(codes, weights=w, minlength=outcomes_arr.shape[0])

    if not np.isfinite(probs_arr).all():
        raise WeightedPMFError(
            f"non-finite calibrated probability for player_id={label.player_id!r} "
            f"prop_type={prop.value!r}"
        )
    if (probs_arr <= 0.0).any():
        raise WeightedPMFError(
            f"non-positive calibrated probability among counted outcomes for "
            f"player_id={label.player_id!r} prop_type={prop.value!r}"
        )
    total = float(probs_arr.sum())
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise WeightedPMFError(
            f"calibrated PMF for player_id={label.player_id!r} prop_type={prop.value!r} "
            f"sums to {total!r}, not 1.0 within {NORMALIZATION_TOLERANCE}"
        )
    return WeightedPMF(
        player_id=label.player_id,
        prop_type=prop,
        n_draws=game.n_draws,
        support_min=int(outcomes_arr[0]),
        support_max=int(outcomes_arr[-1]),
        outcomes=tuple(int(x) for x in outcomes_arr.tolist()),
        probabilities=tuple(float(p) for p in probs_arr.tolist()),
    )


def compact_first_td_simplex(game: CompactGame, weights: np.ndarray) -> dict[str, float]:
    """`build_weighted_first_td_simplex` from the compact first-TD codes."""
    w = validate_draw_weights(weights, game.n_draws)
    probs = np.bincount(game.first_td_codes, weights=w, minlength=len(game.first_td_labels))
    total = float(probs.sum())
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise WeightedPMFError(
            f"weighted first_td simplex sums to {total!r}, not 1.0 within "
            f"{NORMALIZATION_TOLERANCE}"
        )
    if not np.isfinite(probs).all() or (probs <= 0.0).any():
        raise WeightedPMFError("weighted first_td simplex has a non-finite or non-positive mass")
    return dict(zip(game.first_td_labels, (float(p) for p in probs.tolist()), strict=True))


# ------------------------------------------------------------- persistence


def _meta(game: CompactGame) -> dict[str, Any]:
    return {
        "version": COMPACT_GAME_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "game_id": game.game_id,
        "as_of": game.as_of.isoformat(),
        "outcome_available_at": game.outcome_available_at.isoformat(),
        "injury_data_available": game.injury_data_available,
        "n_draws": game.n_draws,
        "model_version": game.model_version,
        "labels": [
            [label.player_id, label.prop_type.value, repr(label.observed_value)]
            for label in game.labels
        ],
        "position_groups": list(game.position_groups),
        "supports": [s.tolist() for s in game.supports],
        "first_td_labels": list(game.first_td_labels),
        "codes_dtype": str(game.codes.dtype),
        "first_td_dtype": str(game.first_td_codes.dtype),
    }


def compact_game_sha256(game: CompactGame) -> str:
    """Canonical content hash: metadata (incl. labels and supports) plus
    the exact little-endian bytes of every array."""
    digest = hashlib.sha256()
    digest.update(json.dumps(_meta(game), sort_keys=True, separators=(",", ":")).encode())
    for array in (game.features, game.codes, game.first_td_codes):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype.newbyteorder("<")).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def write_compact_game(game: CompactGame, directory: Path) -> str:
    """Write one game's compact evidence into `directory/<game_id>/`
    (written to a temp dir, then renamed: a partial game never appears).
    Returns its content hash."""
    target = directory / game.game_id
    if target.exists():
        raise CompactGameError(f"compact game {game.game_id!r} already written (immutable)")
    tmp = directory / f".{game.game_id}.partial"
    tmp.mkdir(parents=True, exist_ok=False)
    meta = _meta(game)
    meta["sha256"] = compact_game_sha256(game)
    np.save(tmp / _FEATURES, game.features)
    np.save(tmp / _CODES, game.codes)
    np.save(tmp / _FIRST_TD, game.first_td_codes)
    (tmp / _META).write_text(json.dumps(meta, sort_keys=True, indent=1))
    tmp.rename(target)
    return str(meta["sha256"])


def read_compact_game(path: Path, *, mmap: bool = True, verify: bool = True) -> CompactGame:
    """Load one compact game. Arrays are memory-mapped by default, so a
    process holding every game keeps only file-backed (reclaimable) pages."""
    from nflprops.calibration.challenger import PropLabel

    meta = json.loads((path / _META).read_text())
    if meta.get("version") != COMPACT_GAME_VERSION:
        raise CompactGameError(f"{path}: unsupported compact game version {meta.get('version')!r}")
    if meta["feature_names"] != list(FEATURE_NAMES):
        raise CompactGameError(f"{path}: feature contract mismatch")
    mode: Literal["r"] | None = "r" if mmap else None
    game = CompactGame(
        game_id=meta["game_id"],
        as_of=datetime.fromisoformat(meta["as_of"]),
        outcome_available_at=datetime.fromisoformat(meta["outcome_available_at"]),
        injury_data_available=bool(meta["injury_data_available"]),
        labels=tuple(
            PropLabel(player_id=pid, prop_type=PropType(prop), observed_value=float(value))
            for pid, prop, value in meta["labels"]
        ),
        n_draws=int(meta["n_draws"]),
        model_version=meta["model_version"],
        features=np.load(path / _FEATURES, mmap_mode=mode),
        supports=tuple(np.asarray(s, dtype=np.int64) for s in meta["supports"]),
        codes=np.load(path / _CODES, mmap_mode=mode),
        position_groups=tuple(meta["position_groups"]),
        first_td_labels=tuple(meta["first_td_labels"]),
        first_td_codes=np.load(path / _FIRST_TD, mmap_mode=mode),
    )
    if verify and compact_game_sha256(game) != meta["sha256"]:
        raise CompactGameError(f"{path}: content hash mismatch")
    return game

