"""PR #21: controlled `games` backfill from genuine raw BDL receipts.

Raw/provider-derived rows only; genuine receipt timestamps only (never
estimated); dry-run by default; exact expected-ID guard; idempotent; writer
lock; immutable snapshots unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nflprops.data.raw_store import RawStore
from nflprops.data.warehouse import Warehouse
from nflprops.domain.ids import canonical_game_id
from nflprops.platform.games_backfill import (
    GAMES_RECEIPT_DIR,
    GamesBackfillError,
    apply_games_backfill,
    plan_games_backfill,
)
from nflprops.platform.immutable_bundle import (
    read_manifest,
    verify_directory_against_manifest,
)
from nflprops.platform.science_readiness import check_science_readiness
from nflprops.platform.warehouse_snapshot import create_snapshot, restore_snapshot
from nflprops.providers.bdl import endpoints

SEASON = 2026
FIRST = datetime(2026, 9, 24, 23, 16, 47, 654142, tzinfo=UTC)
SECOND = FIRST + timedelta(days=1)
AS_OF = datetime(2026, 10, 5, 22, 45, tzinfo=UTC)
NOW = AS_OF + timedelta(days=2)
W1_W2 = {101: 1, 102: 1, 201: 2, 202: 2}
W3 = 301


def _team(team_id: int) -> dict:
    return {"id": team_id, "conference": "AFC", "division": "EAST", "location": f"L{team_id}",
            "name": f"N{team_id}", "full_name": f"Team {team_id}", "abbreviation": f"T{team_id}"}


def _game(game_id: int, week: int, *, status: str = "Final") -> dict:
    return {
        "id": game_id, "season": SEASON, "week": week, "postseason": False,
        "date": f"2026-09-{10 + week:02d}T17:00:00.000Z", "status": status,
        "home_team": _team(1), "visitor_team": _team(2),
        "home_team_score": 21, "visitor_team_score": 17,
    }


def _write_receipt(raw_root: Path, games: list[dict], received_at: datetime, **params) -> str:
    ref = RawStore(raw_root).write_json(
        provider="balldontlie", endpoint=endpoints.GAMES,
        request_params={"per_page": 100, "seasons[]": SEASON, **params},
        payload={"data": games, "meta": {"per_page": 100}},
        requested_at=received_at - timedelta(milliseconds=100), received_at=received_at,
        http_status=200, spec_sha256=None,
    )
    return ref.response_sha256


def _cid(game_id: int) -> str:
    return canonical_game_id("balldontlie", game_id)


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    raw_root = tmp_path / "data" / "raw"
    # Two genuine whole-season discovery pages (scores change between them)
    # and one weekly page for Week 3.
    season_page = [_game(gid, wk) for gid, wk in W1_W2.items()] + [_game(W3, 3, status="Scheduled")]
    _write_receipt(raw_root, season_page, FIRST, season_type=2)
    later = [dict(g, home_team_score=24) for g in season_page]
    _write_receipt(raw_root, later, SECOND, season_type=2)
    _write_receipt(raw_root, [_game(W3, 3, status="Scheduled")], SECOND, **{"weeks[]": 3})

    warehouse = Warehouse(tmp_path / "data" / "canonical")
    seen = AS_OF - timedelta(days=3)
    # Live games table only knows Week 3 (the production defect).
    warehouse.write("games", pl.DataFrame({
        "canonical_game_id": [_cid(W3)], "season": [SEASON], "week": [3],
        "available_at": [seen], "ingested_at": [seen], "available_at_is_estimated": [False],
        "date": [datetime(2026, 9, 27, 17, tzinfo=UTC)],
    }))
    state_games = [_cid(g) for g in W1_W2] + [_cid(W3)]
    warehouse.write("team_game_stats", pl.DataFrame({
        "canonical_game_id": state_games,
        "canonical_team_id": [f"t{i}" for i in range(len(state_games))],
        "available_at": [seen] * len(state_games),
        "available_at_is_estimated": [False] * len(state_games),
    }))
    return {"raw": raw_root, "warehouse": warehouse, "lock": tmp_path / "writer.lock",
            "expected": tuple(sorted(_cid(g) for g in W1_W2)), "tmp": tmp_path}


def _games_bytes(warehouse: Warehouse) -> bytes:
    return warehouse.table_path("games").read_bytes()


def test_dry_run_plans_genuine_rows_and_writes_nothing(env: dict) -> None:
    before = _games_bytes(env["warehouse"])
    plan = plan_games_backfill(env["warehouse"], env["raw"], season=SEASON, weeks=(1, 2),
                               expected_game_ids=env["expected"])
    assert _games_bytes(env["warehouse"]) == before
    assert plan.rows.height == 8  # 4 games x 2 receipts, each its own PIT version
    assert plan.rows_to_write == 8
    assert set(plan.rows["available_at"].to_list()) == {FIRST, SECOND}
    assert (plan.rows["available_at"] == plan.rows["ingested_at"]).all()
    assert not plan.rows["available_at_is_estimated"].any()
    assert set(plan.rows["week"].to_list()) == {1, 2}
    summary = plan.summary()
    assert len(summary["receipts_used"]) == 2  # the Week-3 page holds no W1/W2 game
    assert summary["receipts_scanned"] == 3
    assert {g["first_received_at"] for g in summary["games"]} == {FIRST.isoformat()}
    # The later receipt's content is kept as a later version, not merged.
    latest = plan.rows.sort("available_at").group_by("canonical_game_id").last()
    assert set(latest["home_team_score"].to_list()) == {24}


def test_apply_makes_the_gate_pass_and_is_idempotent(env: dict) -> None:
    wh = env["warehouse"]
    assert check_science_readiness(wh, scheduled_as_of=AS_OF).missing_game_ids == env["expected"]
    first = apply_games_backfill(wh, env["raw"], season=SEASON, weeks=(1, 2),
                                 expected_game_ids=env["expected"], lock_path=env["lock"], now=NOW)
    assert first["rows_to_write"] == 8
    assert check_science_readiness(wh, scheduled_as_of=AS_OF).ready
    # Week-3 row untouched (keep="first": no existing row is ever replaced).
    w3 = wh.read("games").filter(pl.col("canonical_game_id") == _cid(W3))
    assert w3.height == 1 and w3["available_at"][0] == AS_OF - timedelta(days=3)
    after = _games_bytes(wh)
    second = apply_games_backfill(wh, env["raw"], season=SEASON, weeks=(1, 2),
                                  expected_game_ids=env["expected"], lock_path=env["lock"],
                                  now=NOW)
    assert second["rows_to_write"] == 0 and second["already_present_rows"] == 8
    assert _games_bytes(wh) == after


def test_restored_metadata_is_invisible_before_its_genuine_receipt(env: dict) -> None:
    """No availability is fabricated: a cutoff before the first receipt
    still cannot prove those games, one at/after it can."""
    wh = env["warehouse"]
    stats = wh.read("team_game_stats")
    wh.write("team_game_stats", stats.with_columns(
        pl.lit(FIRST - timedelta(hours=1)).cast(stats.schema["available_at"]).alias("available_at")
    ))
    apply_games_backfill(wh, env["raw"], season=SEASON, weeks=(1, 2),
                         expected_game_ids=env["expected"], lock_path=env["lock"], now=NOW)
    before = check_science_readiness(wh, scheduled_as_of=FIRST - timedelta(microseconds=1))
    assert before.missing_game_ids == tuple(sorted((*env["expected"], _cid(W3))))
    at = check_science_readiness(wh, scheduled_as_of=FIRST)
    assert at.missing_game_ids == (_cid(W3),)  # W3's live row arrives later still


@pytest.mark.parametrize("which", ["subset", "superset", "wrong_week"])
def test_exact_expected_id_guard_refuses_and_writes_nothing(env: dict, which: str) -> None:
    expected = {
        "subset": env["expected"][:3],
        "superset": (*env["expected"], _cid(999)),
        "wrong_week": env["expected"],
    }[which]
    weeks = (1,) if which == "wrong_week" else (1, 2)
    before = _games_bytes(env["warehouse"])
    with pytest.raises(GamesBackfillError, match="exact expected-ID guard"):
        apply_games_backfill(env["warehouse"], env["raw"], season=SEASON, weeks=weeks,
                             expected_game_ids=expected, lock_path=env["lock"], now=NOW)
    assert _games_bytes(env["warehouse"]) == before


def test_refuses_when_a_game_is_not_actually_missing(env: dict) -> None:
    wh = env["warehouse"]
    live = wh.read("games")
    foreign = live.with_columns(pl.lit(env["expected"][0]).alias("canonical_game_id"),
                                pl.lit(1, dtype=live.schema["week"]).alias("week"))
    wh.write("games", pl.concat([live, foreign]))
    with pytest.raises(GamesBackfillError, match="not written by this backfill"):
        plan_games_backfill(wh, env["raw"], season=SEASON, weeks=(1, 2),
                            expected_game_ids=env["expected"])


def test_refuses_a_tampered_payload(env: dict) -> None:
    directory = env["raw"] / GAMES_RECEIPT_DIR
    victim = sorted(directory.glob("*.json.zst"))[0]
    from nflprops.data.raw_store import compress_bytes

    victim.write_bytes(compress_bytes(b'{"data":[]}\n'))
    with pytest.raises(GamesBackfillError, match="does not match its address"):
        plan_games_backfill(env["warehouse"], env["raw"], season=SEASON, weeks=(1, 2),
                            expected_game_ids=env["expected"])


@pytest.mark.parametrize("received_at", [None, "2026-09-24T23:16:47"])
def test_refuses_a_receipt_without_a_genuine_timezone_aware_time(
    env: dict, received_at: str | None
) -> None:
    directory = env["raw"] / GAMES_RECEIPT_DIR
    meta_path = sorted(directory.glob("*.meta.json"))[0]
    meta = json.loads(meta_path.read_text())
    meta["received_at"] = received_at
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(GamesBackfillError, match="received_at"):
        plan_games_backfill(env["warehouse"], env["raw"], season=SEASON, weeks=(1, 2),
                            expected_game_ids=env["expected"])


def test_refuses_a_receipt_stamped_after_now(env: dict) -> None:
    with pytest.raises(GamesBackfillError, match="after now"):
        apply_games_backfill(env["warehouse"], env["raw"], season=SEASON, weeks=(1, 2),
                             expected_game_ids=env["expected"], lock_path=env["lock"],
                             now=FIRST)


def test_existing_immutable_snapshot_is_unchanged_and_still_refuses(env: dict) -> None:
    wh = env["warehouse"]
    snapshots = env["tmp"] / "snapshots"
    info = create_snapshot(warehouse_root=wh.root, snapshot_root=snapshots,
                           lock_path=env["lock"], migration_head="h", hostname="h")
    snap_dir = snapshots / info.snapshot_id
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in snap_dir.rglob("*") if p.is_file()}

    apply_games_backfill(wh, env["raw"], season=SEASON, weeks=(1, 2),
                         expected_game_ids=env["expected"], lock_path=env["lock"], now=NOW)

    assert {p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snap_dir.rglob("*") if p.is_file()} == digests
    verify_directory_against_manifest(snap_dir, read_manifest(snap_dir),
                                      expected_manifest_sha256=info.manifest_sha256)
    restored = Warehouse(env["tmp"] / "restored")
    restore_snapshot(snapshots, info.snapshot_id, restored.root)
    old = check_science_readiness(restored, scheduled_as_of=AS_OF)
    assert not old.ready and old.missing_game_ids == env["expected"]
    assert check_science_readiness(wh, scheduled_as_of=AS_OF).ready


def test_cli_is_dry_run_by_default(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from nflprops.platform import wizard_runtime
    from nflprops.platform.runtime_layout import resolve_runtime_layout

    layout = resolve_runtime_layout(env["warehouse"].root, {})
    monkeypatch.setattr(wizard_runtime, "_layout", lambda: layout)
    before = _games_bytes(env["warehouse"])
    args = ["games-backfill", "--season", str(SEASON), "--weeks", "1,2",
            "--expected-ids", ",".join(env["expected"])]
    dry = CliRunner().invoke(wizard_runtime.app, args)
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["dry_run"] is True
    assert _games_bytes(env["warehouse"]) == before
    applied = CliRunner().invoke(wizard_runtime.app, [*args, "--apply"])
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["rows_to_write"] == 8
    assert check_science_readiness(env["warehouse"], scheduled_as_of=AS_OF).ready


def test_ops_exposes_a_guarded_dry_run_by_default_operation() -> None:
    ops = (Path(__file__).resolve().parents[2] / "deploy/wizard/ops.sh").read_text()
    block = ops.split("    games-backfill)")[1].split(";;")[0]
    assert "4th argument must be the literal 'apply'" in block
    assert block.count("--apply") == 1
    assert "^[0-9a-f-]{36}(,[0-9a-f-]{36})*$" in block
