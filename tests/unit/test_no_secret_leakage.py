"""Foundation acceptance test: API keys are redacted from log output."""
import logging

from nflprops.logging_ import configure


def test_no_secret_leakage(monkeypatch, capsys):
    monkeypatch.setenv("BDL_API_KEY", "super-secret-key")
    configure("run-1", "INFO")
    logging.getLogger("test").info("key=%s", "super-secret-key")
    captured = capsys.readouterr()
    assert "super-secret-key" not in captured.err
    assert "[REDACTED]" in captured.err
