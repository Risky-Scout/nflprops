"""Foundation acceptance test: deterministic layering and unknown-key rejection."""
import pytest

from nflprops.config import config_sha256, load


def test_config_layering(monkeypatch):
    monkeypatch.setenv("NFLPROPS_LOG_LEVEL", "DEBUG")
    cfg = load(cli_overrides={"simulation.min_draws": 25000})
    assert cfg.get_path("run.log_level") == "DEBUG"
    assert cfg.get_path("simulation.min_draws") == 25000
    assert cfg.get_path("model.version") == "2026.1.0"
    assert config_sha256(cfg) == config_sha256(cfg)
    with pytest.raises(KeyError):
        load(cli_overrides={"simulation.typo_key": 1})
