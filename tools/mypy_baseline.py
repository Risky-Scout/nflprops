"""Audit strict-mypy debt without permitting regressions.

The baseline records normalized mypy findings as:
    (source path, error code, message, count)

Line numbers are deliberately excluded from finding identity so harmless source
movement does not create false regressions. New or changed findings fail CI.
Resolved findings also fail CI until the checked-in baseline is reduced.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ERROR_RE = re.compile(
    r"^(?P<path>src/[^:]+):"
    r"(?P<line>\d+)"
    r"(?::(?P<column>\d+))?: "
    r"error: "
    r"(?P<message>.*?)"
    r"(?:  \[(?P<code>[^\]]+)\])?$"
)

BASELINE_VERSION = 1


class Finding(NamedTuple):
    path: str
    code: str
    message: str


def run_mypy() -> tuple[int, str]:
    process = subprocess.run(
        [sys.executable, "-m", "mypy"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    return process.returncode, process.stdout


def parse_findings(output: str) -> Counter[Finding]:
    findings: Counter[Finding] = Counter()

    for raw_line in output.splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = ERROR_RE.match(line)

        if match is None:
            continue

        finding = Finding(
            path=match.group("path"),
            code=match.group("code") or "unclassified",
            message=match.group("message").strip(),
        )

        findings[finding] += 1

    return findings


def payload(findings: Counter[Finding]) -> dict[str, object]:
    rows = [
        {
            "path": finding.path,
            "code": finding.code,
            "message": finding.message,
            "count": count,
        }
        for finding, count in sorted(
            findings.items(),
            key=lambda item: (
                item[0].path,
                item[0].code,
                item[0].message,
            ),
        )
    ]

    return {
        "version": BASELINE_VERSION,
        "mypy_target_python": "3.11",
        "total_findings": sum(findings.values()),
        "findings": rows,
    }


def load_baseline(path: Path) -> Counter[Finding]:
    data = json.loads(path.read_text())

    if data.get("version") != BASELINE_VERSION:
        raise ValueError(
            f"unsupported mypy baseline version: {data.get('version')!r}"
        )

    rows = data.get("findings")

    if not isinstance(rows, list):
        raise ValueError("mypy baseline findings must be a list")

    findings: Counter[Finding] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid mypy baseline finding")

        finding = Finding(
            path=str(row["path"]),
            code=str(row["code"]),
            message=str(row["message"]),
        )
        findings[finding] += int(row["count"])

    declared_total = int(data.get("total_findings", -1))
    actual_total = sum(findings.values())

    if declared_total != actual_total:
        raise ValueError(
            "mypy baseline total_findings does not match finding counts"
        )

    return findings


def print_delta(
    heading: str,
    delta: Counter[Finding],
) -> None:
    if not delta:
        return

    print()
    print(heading)

    for finding, count in sorted(
        delta.items(),
        key=lambda item: (
            item[0].path,
            item[0].code,
            item[0].message,
        ),
    ):
        print(
            f"{count}x {finding.path} "
            f"[{finding.code}] {finding.message}"
        )


def generate(path: Path) -> int:
    returncode, output = run_mypy()

    if returncode not in {0, 1}:
        print(output)
        print(
            f"mypy infrastructure failure: exit code {returncode}",
            file=sys.stderr,
        )
        return 2

    findings = parse_findings(output)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload(findings),
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    print(f"MYPY_BASELINE_GENERATED={path}")
    print(f"MYPY_BASELINE_FINDINGS={sum(findings.values())}")

    return 0


def check(path: Path) -> int:
    if not path.exists():
        print(f"mypy baseline does not exist: {path}", file=sys.stderr)
        print(
            "Generate the audited baseline with the manual GitHub workflow.",
            file=sys.stderr,
        )
        return 2

    baseline = load_baseline(path)
    returncode, output = run_mypy()

    if returncode not in {0, 1}:
        print(output)
        print(
            f"mypy infrastructure failure: exit code {returncode}",
            file=sys.stderr,
        )
        return 2

    current = parse_findings(output)

    new_findings = current - baseline
    resolved_findings = baseline - current

    print(f"MYPY_CURRENT_FINDINGS={sum(current.values())}")
    print(f"MYPY_BASELINE_FINDINGS={sum(baseline.values())}")
    print(f"MYPY_NEW_FINDINGS={sum(new_findings.values())}")
    print(
        "MYPY_RESOLVED_NOT_REMOVED_FROM_BASELINE="
        f"{sum(resolved_findings.values())}"
    )

    print_delta("NEW_OR_CHANGED_FINDINGS", new_findings)
    print_delta(
        "RESOLVED_FINDINGS_REQUIRING_BASELINE_REDUCTION",
        resolved_findings,
    )

    if new_findings:
        print("MYPY_BASELINE_GATE=FAIL_NEW_DEBT")
        return 1

    if resolved_findings:
        print("MYPY_BASELINE_GATE=FAIL_STALE_BASELINE")
        return 1

    print("MYPY_BASELINE_GATE=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("generate", "check"),
    )
    parser.add_argument(
        "baseline",
        type=Path,
    )
    args = parser.parse_args()

    if args.mode == "generate":
        return generate(args.baseline)

    return check(args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
