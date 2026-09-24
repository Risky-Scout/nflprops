#!/usr/bin/env bash
# BLOCK 3: fixed, auditable operations against the live Wizard runtime.
# Streamed over SSH by .github/workflows/wizard-ops.yml (main-only,
# wizardofodds.com environment):
#
#   ssh ... bash -s -- OPERATION RUNTIME_ROOT [ARGS...] < deploy/wizard/ops.sh
#
# Operations (nothing else is accepted):
#   status                 READ-ONLY: service/process/RAM/swap/disk/clock/
#                          listeners/unrelated-service state + runtime status
#                          + full platform health (reported, not gated)
#   report                 READ-ONLY: live-warehouse certification report
#   collect-once           ONE explicit MANUAL collection cycle
#   manual-checkpoint GAME_ID AS_OF SEASON WEEK
#                          claim + PREPARE one MANUAL checkpoint (no science)
#   verify-snapshot SNAPSHOT_ID
#                          READ-ONLY: re-verify one immutable snapshot
#   restart                sudo -n systemctl restart nflprops-runtime.service,
#                          then GATE on 2 consecutive healthy probes
#
# Runs as wizard-deploy. Touches no other service, nginx, or any path
# outside RUNTIME_ROOT; the only privileged action is the scoped restart.
# Never runs training, replay, simulation, or calibration.

set -uo pipefail

UNIT=nflprops-runtime.service

die() {
  echo "::error::OPS FAILED: $*" >&2
  exit 1
}

section() {
  printf '\n===== %s =====\n' "$*"
}

# Reporting helper for READ-ONLY operations: shows the command and its
# exit status; a non-zero status is reported, never hidden, but does not
# abort the rest of the read-only report.
show() {
  echo "\$ $*"
  "$@"
  echo "[exit $?]"
}

runtime_python() {
  (
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    cd "$ROOT/current" || exit 1
    "$ROOT/current/.venv/bin/python" "$@"
  )
}

op_status() {
  local pid
  section "service"
  show systemctl is-active "$UNIT"
  show systemctl is-enabled "$UNIT"
  show systemctl show "$UNIT" -p MainPID -p NRestarts -p ActiveEnterTimestamp \
    -p MemoryCurrent -p CPUUsageNSec -p ExecMainStatus -p Result
  pid="$(systemctl show "$UNIT" -p MainPID --value)"
  section "process"
  if [ -n "$pid" ] && [ "$pid" != "0" ]; then
    show ps -o pid,rss,vsz,pcpu,etime,nlwp,cmd -p "$pid"
  else
    echo "no MainPID (service not running)"
  fi
  section "memory"
  show free -h
  show swapon --show
  section "disk"
  show df -h /home/wizard-deploy
  for p in "$ROOT/state/canonical" "$ROOT/state/raw" "$ROOT/state/nflprops.duckdb" \
           "$ROOT/snapshots" "$ROOT/publications" "$ROOT/releases" "$ROOT/current/.venv"; do
    if [ -e "$p" ]; then show du -sh "$p"; else echo "absent: $p"; fi
  done
  show ls -1 "$ROOT/releases"
  section "clock"
  show timedatectl status
  show timedatectl show -p NTPSynchronized --value
  section "listeners (read-only)"
  show ss -tlnp
  section "unrelated workloads (read-only)"
  show systemctl --failed --no-pager
  for p in /home/wizard-deploy/nfl-production-2026 /var/www/sportsodds; do
    if [ -e "$p" ]; then show stat -c '%n mtime=%y owner=%U' "$p"; else echo "absent: $p"; fi
  done
  section "runtime status"
  show runtime_python -m nflprops.platform.wizard_runtime status
  show runtime_python -m nflprops.platform.wizard_runtime lock-status
  show runtime_python -m nflprops.platform.wizard_runtime checkpoint pending
  section "platform health (full, reported)"
  show runtime_python -m nflprops.cli platform health
  section "recent runtime journal"
  if journalctl -u "$UNIT" -n 40 --no-pager >/dev/null 2>&1; then
    journalctl -u "$UNIT" -n 40 --no-pager
  else
    echo "journal not readable by $(whoami) -- see $ROOT/logs/runtime-status.json above"
  fi
}

op_restart() {
  local expect passes=0 attempt=0 output
  expect="$(cat "$ROOT/current/RELEASE_SHA")" || die "no RELEASE_SHA in current"
  sudo -n /usr/bin/systemctl restart "$UNIT" || die "restart $UNIT failed"
  while [ "$attempt" -lt 16 ]; do
    attempt=$((attempt + 1))
    sleep 15
    if systemctl is-active --quiet "$UNIT" \
      && output="$(runtime_python -m nflprops.cli platform health --deploy-gate --expect-version "$expect" 2>&1)"; then
      passes=$((passes + 1))
      echo "probe $attempt passed ($passes/2)"
      if [ "$passes" -ge 2 ]; then
        echo "RESTARTED: $UNIT healthy on $expect"
        return 0
      fi
    else
      passes=0
      echo "probe $attempt failed" >&2
      printf '%s\n' "${output:-}" | tail -n 60 >&2
    fi
  done
  die "$UNIT did not become healthy after restart"
}

main() {
  [ "$#" -ge 2 ] || die "usage: OPERATION RUNTIME_ROOT [ARGS...]"
  local op="$1"
  ROOT="$2"
  shift 2
  case "$ROOT" in
    /home/wizard-deploy/nflprops) ;;
    *) die "RUNTIME_ROOT must be /home/wizard-deploy/nflprops" ;;
  esac
  ENV_FILE="$ROOT/nflprops-runtime.env"
  [ -f "$ENV_FILE" ] || die "$ENV_FILE missing"
  [ -x "$ROOT/current/.venv/bin/python" ] || die "no active release"

  case "$op" in
    status) op_status ;;
    report) runtime_python -m nflprops.platform.wizard_runtime report || die "report failed" ;;
    collect-once) runtime_python -m nflprops.platform.wizard_runtime once || die "collect-once failed" ;;
    manual-checkpoint)
      [ "$#" -eq 4 ] || die "manual-checkpoint needs GAME_ID AS_OF SEASON WEEK"
      runtime_python -m nflprops.platform.wizard_runtime checkpoint manual \
        --game-id "$1" --as-of "$2" --season "$3" --week "$4" || die "manual checkpoint failed"
      ;;
    verify-snapshot)
      [ "$#" -eq 1 ] || die "verify-snapshot needs SNAPSHOT_ID"
      runtime_python -m nflprops.platform.wizard_runtime snapshot verify "$1" || die "snapshot verify failed"
      ;;
    restart) op_restart ;;
    *) die "unknown operation $op" ;;
  esac
}

main "$@" </dev/null
