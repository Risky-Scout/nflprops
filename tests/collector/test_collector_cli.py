"""PHASE 4: provider-neutral `nflprops collect` CLI commands.

Never touches the real BDL API or the user's authoritative local warehouse --
a fake provider is registered under a test-only name and routed to a
temporary warehouse, exercising the exact same `registry.get_provider(...)`
path production code uses.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider_contract"))

from fake_provider import FakeProvider
from typer.testing import CliRunner

from nflprops.cli import app
from nflprops.data.warehouse import Warehouse
from nflprops.providers import registry

runner = CliRunner()
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _make_seeded_provider() -> FakeProvider:
    provider = FakeProvider()
    provider.seed_team("t1", nickname="Home", abbreviation="HOM")
    provider.seed_team("t2", nickname="Away", abbreviation="AWY")
    provider.seed_player("p1", first_name="Home", last_name="Player")
    provider.seed_game(
        "g1",
        home_team_native_id="t1",
        visitor_team_native_id="t2",
        week=1,
        date=NOW + timedelta(hours=5),
    )
    return provider


def test_collect_once_runs_against_a_registered_test_provider(tmp_path: Path) -> None:
    warehouse = Warehouse(tmp_path / "warehouse")
    registry.register(
        "test-cli-fake-provider", lambda cfg: (_make_seeded_provider(), warehouse)
    )

    result = runner.invoke(
        app, ["collect", "once", "2026", "1", "--provider", "test-cli-fake-provider"]
    )

    assert result.exit_code == 0, result.output
    assert "collector_run_id=" in result.output
    assert warehouse.exists("collector_runs")
    assert warehouse.exists("games")


def test_collect_once_unknown_provider_fails_explicitly() -> None:
    result = runner.invoke(
        app, ["collect", "once", "2026", "1", "--provider", "definitely-not-registered-xyz"]
    )

    assert result.exit_code != 0
    combined = (result.output or "") + str(result.exception or "")
    assert "unknown provider" in combined or "definitely-not-registered-xyz" in combined


def test_collect_once_never_writes_outside_the_returned_warehouse(tmp_path: Path) -> None:
    """The CLI command must not construct or touch any warehouse other than
    the one the registered provider factory returns."""
    warehouse_a = Warehouse(tmp_path / "warehouse-a")
    registry.register(
        "test-cli-fake-provider-2", lambda cfg: (_make_seeded_provider(), warehouse_a)
    )

    runner.invoke(app, ["collect", "once", "2026", "1", "--provider", "test-cli-fake-provider-2"])

    assert warehouse_a.exists("collector_runs")
    assert not (tmp_path / "warehouse-b").exists()


def test_collect_command_group_is_registered() -> None:
    result = runner.invoke(app, ["collect", "--help"])
    assert result.exit_code == 0
    assert "once" in result.output
    assert "loop" in result.output
