"""Typer command tree.

SPEC: docs/IMPLEMENTATION_SPEC.md §73
PHASE: 0
STATUS: PARTIALLY IMPLEMENTED — ingestion/coverage/predict/settle/report are executable
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import UTC, datetime

import typer

from nflprops.paths import repository_root, runtime_resource
from nflprops.providers.bdl.spec import DEFAULT_SPEC_URL, pin_spec
from nflprops.providers.bdl.spec import drift as bdl_spec_drift

app = typer.Typer(
    name="nflprops",
    help="NFL player prop prediction system. Every prop from one simulation.",
    no_args_is_help=True,
)

provider_app = typer.Typer(help="Provider spec pinning, verification, drift.")
ingest_app = typer.Typer(help="Data ingestion.")
snapshot_app = typer.Typer(help="Point-in-time snapshot collection.")
pbp_app = typer.Typer(help="Play-by-play parsing and reconciliation.")
features_app = typer.Typer(help="Point-in-time feature building.")
state_app = typer.Typer(help="Empirical-Bayes state updates.")
collect_app = typer.Typer(help="Continuous point-in-time collection (PHASE 4).")

app.add_typer(provider_app, name="provider")
app.add_typer(ingest_app, name="ingest")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(pbp_app, name="pbp")
app.add_typer(features_app, name="features")
app.add_typer(state_app, name="state")
app.add_typer(collect_app, name="collect")


# --- provider ---------------------------------------------------------------
@provider_app.command("pin")
def provider_pin(provider: str, url: str = DEFAULT_SPEC_URL) -> None:
    """Download and atomically pin a provider OpenAPI specification."""
    if provider != "bdl":
        raise typer.BadParameter("only the bdl provider is currently implemented")
    root = repository_root()
    target = (
        root / "specs" / "providers" / provider / "nfl.yml"
        if root is not None
        else runtime_resource("specs", "providers", provider, "nfl.yml")
    )
    lock = pin_spec(url, target)
    # Keep the installable-package resource mirror synchronized in a checkout.
    if root is not None:
        packaged = root / "src" / "nflprops" / "resources" / "specs" / "providers" / provider
        packaged.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, packaged / "nfl.yml")
        lock_file = target.parent / "spec.lock.json"
        if lock_file.exists():
            shutil.copy2(lock_file, packaged / "spec.lock.json")
    typer.echo(f"Pinned {provider} spec: {target}")
    typer.echo(f"sha256={lock['sha256']}")


@provider_app.command("verify")
def provider_verify(provider: str, strict_fields: bool = False) -> None:
    """Verify machine-readable endpoint/field contracts against the pinned spec."""
    root = repository_root()
    if root is None:
        raise typer.BadParameter(
            "provider verification is a repository-development command; run it from a clone"
        )
    tool = root / "tools" / "verify_spec_coverage.py"
    cmd = [sys.executable, str(tool), "--provider", provider]
    if strict_fields:
        cmd.append("--strict-fields")
    result = subprocess.run(cmd, cwd=root)
    if result.returncode:
        raise typer.Exit(result.returncode)


@provider_app.command("drift")
def provider_drift(provider: str) -> None:
    """Compare the pinned spec with upstream; never modify the production pin."""
    if provider != "bdl":
        raise typer.BadParameter("only the bdl provider is currently implemented")
    target = runtime_resource("specs", "providers", provider, "nfl.yml")
    if not target.exists():
        raise typer.BadParameter("no pinned spec exists")
    pinned_sha, live_sha = bdl_spec_drift(target)
    typer.echo(f"pinned_sha256={pinned_sha}")
    typer.echo(f"live_sha256={live_sha}")
    if pinned_sha != live_sha:
        typer.echo("DRIFT DETECTED — production pin was NOT modified.")
        raise typer.Exit(2)
    typer.echo("No provider-spec drift detected.")


# --- ingest -----------------------------------------------------------------
@ingest_app.command("bootstrap")
def ingest_bootstrap(provider: str = "bdl") -> None:
    """Fetch reference teams/players into the lean local warehouse."""
    from nflprops.config import load
    from nflprops.pipelines.lean import LeanIngestor
    from nflprops.providers.registry import get_provider

    cfg = load(provider=provider)
    try:
        source, warehouse = get_provider(provider, cfg)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        LeanIngestor(
            source,
            warehouse,
            goat=bool(cfg.get_path("provider.bdl.tier.goat", False)),
        ).bootstrap()
    finally:
        source.client.close()
    typer.echo("Reference data ingested.")


@ingest_app.command("season")
def ingest_season(
    season: int,
    include_pbp: bool = typer.Option(False, help="Also ingest play-by-play; not required for Tier-1 full-game props."),
) -> None:
    """Backfill one season into canonical Parquet/DuckDB tables."""
    from nflprops.config import load
    from nflprops.pipelines.lean import LeanIngestor
    from nflprops.providers.registry import get_provider

    cfg = load()
    source, warehouse = get_provider("bdl", cfg)
    try:
        LeanIngestor(
            source,
            warehouse,
            goat=bool(cfg.get_path("provider.bdl.tier.goat", False)),
        ).ingest_season(
            season,
            include_pbp=include_pbp,
            historical_backfill=True,
        )
    finally:
        source.client.close()
    typer.echo(f"Season {season} ingested.")


@ingest_app.command("week")
def ingest_week(season: int, week: int) -> None:
    """Refresh the current week: games, odds, injuries, rosters, player props."""
    from nflprops.config import load
    from nflprops.pipelines.lean import LeanIngestor
    from nflprops.providers.registry import get_provider

    cfg = load()
    source, warehouse = get_provider("bdl", cfg)
    try:
        LeanIngestor(
            source,
            warehouse,
            goat=bool(cfg.get_path("provider.bdl.tier.goat", False)),
        ).ingest_week(season, week)
    finally:
        source.client.close()
    typer.echo(f"Season {season} week {week} refreshed.")


# --- collection (PHASE 4) ----------------------------------------------------
@collect_app.command("once")
def collect_once_cmd(
    season: int,
    week: int,
    provider: str = "bdl",
) -> None:
    """Run exactly one collection cycle: schedule, rosters, injuries, odds, props."""
    from nflprops.collection.service import collect_once
    from nflprops.config import load
    from nflprops.pipelines.lean import LeanIngestor  # noqa: F401 -- registers "bdl"
    from nflprops.providers.registry import get_provider

    cfg = load()
    try:
        source, warehouse = get_provider(provider, cfg)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        result = collect_once(
            provider=source,
            season=season,
            week=week,
            warehouse=warehouse,
            config=cfg,
            now=datetime.now(UTC),
        )
    finally:
        client = getattr(source, "client", None)
        if client is not None:
            client.close()
    typer.echo(
        f"collector_run_id={result.collector_run_id} status={result.status.value} "
        f"cadence_seconds={result.cadence_seconds} "
        f"games={result.games_received} odds_rows={result.game_odds_rows} "
        f"prop_rows={result.prop_rows} roster_rows={result.roster_rows} "
        f"injury_rows={result.injury_rows}"
    )


@collect_app.command("loop")
def collect_loop_cmd(
    season: int,
    week: int,
    provider: str = "bdl",
) -> None:
    """Run collection cycles continuously (foreground) until interrupted."""
    from nflprops.collection.loop import run_collection_loop
    from nflprops.config import load
    from nflprops.pipelines.lean import LeanIngestor  # noqa: F401 -- registers "bdl"
    from nflprops.providers.registry import get_provider

    cfg = load()
    try:
        source, warehouse = get_provider(provider, cfg)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    def _report(result: object) -> None:
        typer.echo(
            f"collector_run_id={result.collector_run_id} status={result.status.value} "
            f"cadence_seconds={result.cadence_seconds}"
        )

    try:
        run_collection_loop(
            provider=source,
            season=season,
            week=week,
            warehouse=warehouse,
            config=cfg,
            on_cycle=_report,
        )
    finally:
        client = getattr(source, "client", None)
        if client is not None:
            client.close()


# --- snapshots --------------------------------------------------------------
@snapshot_app.command("injuries")
def snapshot_injuries() -> None:
    """PHASE 2."""
    raise NotImplementedError("PHASE 2")


@snapshot_app.command("rosters")
def snapshot_rosters() -> None:
    """PHASE 2."""
    raise NotImplementedError("PHASE 2")


@snapshot_app.command("odds")
def snapshot_odds() -> None:
    """PHASE 9."""
    raise NotImplementedError("PHASE 9")


@snapshot_app.command("props")
def snapshot_props() -> None:
    """Collect live player props.

    START THIS AS SOON AS PHASE 1 LANDS. BDL retains no historical live prop data;
    every uncollected week is permanently lost. SPEC §58. PHASE 9.
    """
    raise NotImplementedError("PHASE 9")


# --- pbp --------------------------------------------------------------------
@pbp_app.command("parse")
def pbp_parse(season: int) -> None:
    """PHASE 3."""
    raise NotImplementedError("PHASE 3")


@pbp_app.command("reconcile")
def pbp_reconcile(season: int) -> None:
    """PHASE 3."""
    raise NotImplementedError("PHASE 3")


# --- features / state -------------------------------------------------------
@features_app.command("build")
def features_build(as_of: str) -> None:
    """PHASE 4."""
    raise NotImplementedError("PHASE 4")


@state_app.command("update")
def state_update(as_of: str) -> None:
    """PHASE 5."""
    raise NotImplementedError("PHASE 5")


# --- modelling / execution --------------------------------------------------
@app.command("train")
def train(cutoff: str) -> None:
    """PHASE 6."""
    raise NotImplementedError("PHASE 6")


@app.command("backtest")
def backtest(seasons: str) -> None:
    """PHASE 10."""
    raise NotImplementedError("PHASE 10")


@app.command("predict")
def predict(
    season: int,
    week: int,
    as_of: str,
    draws: int = typer.Option(20000, min=1000),
) -> None:
    """Build PIT states, simulate each game once, and price current player props."""
    from nflprops.config import load
    from nflprops.pipelines.lean import open_warehouse
    from nflprops.pipelines.pregame import (
        predict_week,
        simulation_config_from_app_config,
        state_configs_from_app_config,
    )

    dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise typer.BadParameter("as_of must include an explicit timezone")
    cfg = load()
    warehouse = open_warehouse(cfg)
    player_state_cfg, team_state_cfg = state_configs_from_app_config(cfg)
    out = predict_week(
        warehouse,
        season=season,
        week=week,
        as_of=dt,
        model_version=str(cfg.get_path("model.version", "2026.1.0")),
        n_draws=draws,
        retain_joint_draws=int(cfg.get_path("simulation.retain_joint_draws", 0)),
        simulation_config=simulation_config_from_app_config(cfg, n_draws=draws),
        player_state_config=player_state_cfg,
        team_state_config=team_state_cfg,
        max_confidence_tier=int(cfg.get_path("market.max_confidence_tier", 2)),
    )
    if out.is_empty():
        typer.echo("No priceable prop quotes found.")
        return
    display = out.select(
        "game_id", "player_id", "prop_type", "confidence_tier", "vendor", "side", "line",
        "american_odds", "model_mean", "p_model_raw", "p_market_fair", "edge",
        "ev_per_unit"
    ).sort("ev_per_unit", descending=True)
    typer.echo(display)


@app.command("run")
def run_week(
    season: int,
    week: int,
    as_of: str = typer.Option(
        "",
        help="Explicit ISO timestamp; defaults to the instant after live refresh.",
    ),
    draws: int = typer.Option(20000, min=1000),
) -> None:
    """Refresh BDL current-week inputs and immediately publish lean predictions."""
    ingest_week(season, week)
    stamp = as_of or datetime.now(UTC).isoformat()
    predict(season, week, stamp, draws)


@app.command("settle")
def settle(season: int, week: int) -> None:
    """Settle structured full-game props from BDL player-game stats."""
    import polars as pl

    from nflprops.config import load
    from nflprops.pipelines.lean import open_warehouse
    from nflprops.pipelines.settle import settle_predictions

    warehouse = open_warehouse(load())
    games = warehouse.read("games")
    predictions = warehouse.read("predictions")
    stats = warehouse.read("player_game_stats")
    if games.is_empty() or predictions.is_empty() or stats.is_empty():
        typer.echo("Nothing to settle.")
        return
    game_ids = (
        games.filter((pl.col("season") == season) & (pl.col("week") == week))
        ["canonical_game_id"].unique().to_list()
    )
    preds = predictions.filter(pl.col("game_id").is_in(game_ids))
    settled = settle_predictions(preds, stats)
    if settled.is_empty():
        typer.echo("No structured full-game predictions were settleable.")
        return
    warehouse.append(
        "settled_predictions",
        settled,
        key=["prediction_id"],
        sort_by=["settled_at", "game_id", "player_id"],
    )
    typer.echo(f"Settled {settled.height} prediction sides.")


@app.command("report")
def report(season: int, week: int) -> None:
    """Report same-timestamp model-vs-market scoring for settled non-push rows."""
    import polars as pl

    from nflprops.backtest.metrics import compare_to_market
    from nflprops.config import load
    from nflprops.pipelines.lean import open_warehouse

    warehouse = open_warehouse(load())
    settled = warehouse.read("settled_predictions")
    games = warehouse.read("games")
    if settled.is_empty() or games.is_empty():
        typer.echo("No settled predictions.")
        return
    game_ids = (
        games.filter((pl.col("season") == season) & (pl.col("week") == week))
        ["canonical_game_id"].unique().to_list()
    )
    rows = settled.filter(
        pl.col("game_id").is_in(game_ids)
        & pl.col("outcome_binary").is_not_null()
        & pl.col("p_market_fair").is_not_null()
    )
    if rows.is_empty():
        typer.echo("No same-timestamp two-sided market comparisons for this week.")
        return
    bench = compare_to_market(
        rows["outcome_binary"].to_numpy(),
        rows["p_model_raw"].to_numpy(),
        rows["p_market_fair"].to_numpy(),
    )
    typer.echo(
        f"n={bench.n} model_logloss={bench.model_log_loss:.5f} "
        f"market_logloss={bench.market_log_loss:.5f} "
        f"improvement={bench.log_loss_improvement:+.5f}"
    )
    typer.echo(
        f"model_brier={bench.model_brier:.5f} "
        f"market_brier={bench.market_brier:.5f} "
        f"improvement={bench.brier_improvement:+.5f}"
    )
    typer.echo(
        f"model_ECE={bench.model_ece:.4f} market_ECE={bench.market_ece:.4f}"
    )


@app.command("reproduce")
def reproduce(
    run_id: str,
    artifact_root: str = typer.Option(
        "artifacts/validation",
        help="Root containing immutable validation run directories.",
    ),
) -> None:
    """Rebuild from raw + manifest and assert byte-identical output. SPEC §67."""
    from pathlib import Path

    from nflprops.backtest.reproduce import (
        reproduce_run_directory,
    )

    result = reproduce_run_directory(
        Path(artifact_root) / run_id,
        repo_root=Path.cwd(),
    )

    typer.echo(
        "REPRODUCIBILITY=PASS "
        f"run_id={run_id} "
        f"rows={result.row_count} "
        f"manifest_sha256={result.manifest_sha256} "
        f"probability_sha256={result.first_probability_sha256}"
    )


@app.command("coverage")
def coverage() -> None:
    """Show the empirically ingested local coverage; hardcodes no provider history."""
    from nflprops.config import load
    from nflprops.pipelines.lean import open_warehouse

    warehouse = open_warehouse(load())
    tables = warehouse.tables()
    if not tables:
        typer.echo("No canonical tables have been ingested.")
        return
    for table in tables:
        frame = warehouse.read(table)
        typer.echo(f"{table}: {frame.height} rows")


if __name__ == "__main__":
    app()
