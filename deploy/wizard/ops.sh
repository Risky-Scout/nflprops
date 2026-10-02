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
#   inventory              READ-ONLY: disk/inode usage + per-path physical
#                          inventory of RUNTIME_ROOT only (canonical files,
#                          releases, snapshots + their protection, raw
#                          payload growth, logs/backups, temp/orphan files)
#   restart                sudo -n systemctl restart nflprops-runtime.service,
#                          then GATE on 2 consecutive healthy probes
#   checkpoint-select      READ-ONLY (BLOCK 4): JSON list of executable
#                          PENDING_REMOTE_EXECUTION requests (+ any already
#                          published result bundle SHA) for checkpoint-execute.yml
#   result-ingest BUNDLE_ID MANIFEST_SHA256
#                          BLOCK 4: validate + install ONE published GitHub
#                          result bundle (runtime owner, writer lock; idempotent)
#   checkpoint-refuse RUN_ID WORKFLOW_RUN_URL
#                          BLOCK 4: the executor's verification refused this
#                          request -> NOT_EXECUTABLE (fail closed; idempotent)
#   outcome-ingest-hold    BLOCK 4: create $ROOT/state/outcome_ingest.hold --
#                          the runtime's recurring outcome ingest never starts
#   outcome-ingest-release BLOCK 4: remove that hold file
#   outcome-report [AS_OF] READ-ONLY (BLOCK 4): versioned outcome-table
#                          certification (+ what a cutoff AS_OF could see)
#   ingest-stats SEASON [WEEKS]
#                          BLOCK 4: append final-game outcome versions
#                          (genuine receipt time; never fabricated PIT)
#
# Runs as wizard-deploy. Touches no other service, nginx, or any path
# outside RUNTIME_ROOT; the only privileged action is the scoped restart.
# Never runs training, replay, simulation, or calibration.
# (result-ingest only appends GitHub-produced rows; it computes no science.)

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

# ---------------------------------------------------------------- inventory
# READ-ONLY. Every command below only stats/lists/reads files under $ROOT
# (plus `df` of /). Nothing is created, moved, or deleted.

# "<bytes> <mtime> <path>" for one path (dir sizes are du apparent bytes).
inv_entry() {
  local p="$1" bytes mtime
  bytes="$(du -sb "$p" 2>/dev/null | cut -f1)"
  mtime="$(stat -c '%y' "$p" 2>/dev/null | cut -d. -f1)"
  printf '%14s  %s  %s\n' "${bytes:-?}" "${mtime:-?}" "$p"
}

# bytes, file count, oldest, newest of every regular file under $1.
inv_files_summary() {
  find "$1" -type f -printf '%T@ %s %TY-%Tm-%TdT%TH:%TM\n' 2>/dev/null | sort -n | awk '
    NR == 1 { oldest = $3 }
    { bytes += $2; n += 1; newest = $3 }
    END { printf "bytes=%d files=%d oldest=%s newest=%s\n", bytes, n, (n ? oldest : "-"), (n ? newest : "-") }'
}

op_inventory() {
  local d child
  section "filesystem"
  show df -h /
  show df -i /
  show df -B1 /

  section "top-level sizes (bytes, du apparent)"
  for d in "$ROOT" "$ROOT/state" "$ROOT/state/raw" "$ROOT/state/canonical" \
           "$ROOT/snapshots" "$ROOT/releases" "$ROOT/backups" "$ROOT/logs" \
           "$ROOT/publications" "$ROOT/locks"; do
    if [ -e "$d" ]; then
      printf '%14s  %s\n' "$(du -sb "$d" | cut -f1)" "$d"
      printf '%14s  %s (disk usage)\n' "$(du -sB1 "$d" | cut -f1)" "$d"
    else
      echo "absent: $d"
    fi
  done

  section "immediate children (bytes  mtime  path)"
  for d in "$ROOT" "$ROOT/state" "$ROOT/state/raw" "$ROOT/state/canonical" \
           "$ROOT/snapshots" "$ROOT/releases" "$ROOT/backups" "$ROOT/logs" \
           "$ROOT/publications" "$ROOT/locks"; do
    [ -d "$d" ] || continue
    echo "--- $d"
    find "$d" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | sort -z \
      | while IFS= read -r -d '' child; do inv_entry "$child"; done
  done

  section "canonical physical files (per table / *.parts dir)"
  find "$ROOT/state/canonical" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | sort -z \
    | while IFS= read -r -d '' child; do
        if [ -d "$child" ]; then
          printf '%s  DIR  %s\n' "$child" "$(inv_files_summary "$child")"
        else
          printf '%s  FILE bytes=%s mtime=%s\n' "$child" \
            "$(stat -c '%s' "$child")" "$(stat -c '%y' "$child" | cut -d. -f1)"
        fi
      done

  section "temp / staging / orphan candidates under RUNTIME_ROOT (release venvs excluded)"
  find "$ROOT" -path "$ROOT/releases/*/.venv" -prune -o \( \
      -name '*.tmp' -o -name '*.parquet.tmp' -o -name '*.json.tmp' -o -name '*.partial' \
      -o -name '*.pending' -o -name '*.replaced' -o -name '_incoming' -o -name '_pending' \
      -o -name '.*.tmp-*' -o -name '*canary*' -o -name '*.tar.gz' \) -print0 2>/dev/null \
    | sort -z | while IFS= read -r -d '' child; do inv_entry "$child"; done
  echo "(end of candidates)"
  for d in "$ROOT/publications/_incoming" "$ROOT/snapshots/_pending"; do
    if [ -d "$d" ]; then echo "--- $d"; find "$d" -mindepth 1 -maxdepth 1 -print0 \
      | sort -z | while IFS= read -r -d '' child; do inv_entry "$child"; done; fi
  done

  section "releases"
  local current_target
  current_target=""
  if [ -L "$ROOT/current" ]; then current_target="$(readlink -f "$ROOT/current")"; fi
  echo "current -> ${current_target:-<none>}"
  for d in "$ROOT"/releases/*; do
    [ -e "$d" ] || continue
    local is_current=NO prepared=NO sha="-" rollback=NO
    [ "$(readlink -f "$d")" = "$current_target" ] && is_current=YES
    [ -f "$d/.prepared" ] && prepared=YES
    [ -f "$d/RELEASE_SHA" ] && sha="$(cat "$d/RELEASE_SHA")"
    # activate_release.sh keeps exactly the active release and the one it
    # replaced; that one is the rollback target iff it is health-verifiable.
    if [ "$is_current" = NO ] && [ -d "$d" ] && [ "$prepared" = YES ] && [ "$sha" != "-" ]; then
      rollback=YES
    fi
    printf 'release=%s bytes=%s mtime=%s current=%s rollback_target=%s prepared=%s release_sha=%s\n' \
      "$(basename "$d")" "$(du -sb "$d" | cut -f1)" "$(stat -c '%y' "$d" | cut -d. -f1)" \
      "$is_current" "$rollback" "$prepared" "$sha"
  done

  section "raw payloads by provider/endpoint"
  find "$ROOT/state/raw" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null | sort -z \
    | while IFS= read -r -d '' child; do
        printf '%s  %s\n' "${child#"$ROOT"/state/raw/}" "$(inv_files_summary "$child")"
      done
  section "raw payloads by day written (UTC mtime)"
  find "$ROOT/state/raw" -type f -printf '%TY-%Tm-%Td %s\n' 2>/dev/null \
    | awk '{ b[$1] += $2; n[$1] += 1 } END { for (d in b) printf "%s bytes=%d files=%d\n", d, b[d], n[d] }' \
    | sort
  section "raw payload growth windows"
  for minutes in 360 1440 4320; do
    printf 'last_%sm  ' "$minutes"
    find "$ROOT/state/raw" -type f -mmin "-$minutes" -printf '%s\n' 2>/dev/null \
      | awk '{ b += $1; n += 1 } END { printf "bytes=%d files=%d\n", b, n }'
  done
  printf 'raw_total  %s\n' "$(inv_files_summary "$ROOT/state/raw")"
  section "canonical growth windows (files modified within window; bytes = current size)"
  for minutes in 1440 4320; do
    printf 'last_%sm  ' "$minutes"
    find "$ROOT/state/canonical" -type f -mmin "-$minutes" -printf '%s\n' 2>/dev/null \
      | awk '{ b += $1; n += 1 } END { printf "bytes=%d files=%d\n", b, n }'
  done

  section "logs (every file)"
  [ -d "$ROOT/logs" ] && find "$ROOT/logs" -type f -printf '%14s  %TY-%Tm-%TdT%TH:%TM  %p\n' | sort -k2
  section "backups (every file)"
  if [ -d "$ROOT/backups" ]; then
    find "$ROOT/backups" -type f -printf '%14s  %TY-%Tm-%TdT%TH:%TM  %p\n' | sort -k2
  else
    echo "absent: $ROOT/backups"
  fi
  section "publications (every file)"
  [ -d "$ROOT/publications" ] && find "$ROOT/publications" -type f -printf '%14s  %TY-%Tm-%TdT%TH:%TM  %p\n' | sort -k2

  section "snapshots: retention + request protection; superseded compaction parts"
  runtime_python - "$ROOT" <<'PY' || echo "[snapshot/parts analysis failed]"
import json, re, sys
from pathlib import Path

from nflprops.data.warehouse import read_table
from nflprops.platform.runtime_layout import snapshot_retention
from nflprops.platform.warehouse_snapshot import list_snapshots

root = Path(sys.argv[1])
canonical = root / "state" / "canonical"
snap_root = root / "snapshots"

SEEN = set()  # (dev, inode): hard-linked snapshot files count once overall


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def physical_bytes(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            st = p.stat()
            if (st.st_dev, st.st_ino) not in SEEN:
                SEEN.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total

requests = read_table(canonical, "remote_checkpoint_requests")
try:  # the deployed release's own protection rule
    from nflprops.platform.checkpoint_prepare import SNAPSHOT_PROTECTING_STATES as protecting_states
except ImportError:  # releases before bounded storage
    protecting_states = ("PENDING_REMOTE_EXECUTION", "NOT_EXECUTABLE")
print(f"protecting_states={list(protecting_states)}")
by_snapshot = {}
if requests.height:
    for row in requests.iter_rows(named=True):
        if row["snapshot_id"]:
            by_snapshot.setdefault(row["snapshot_id"], []).append(row["state"])
protected = {sid for sid, states in by_snapshot.items() if any(s in protecting_states for s in states)}

keep = snapshot_retention()
snapshots = list_snapshots(snap_root)
unprotected = [s.snapshot_id for s in snapshots if s.snapshot_id not in protected]
kept_by_policy = set(unprotected[-keep:])
listed = {s.snapshot_id for s in snapshots}
totals = {"all": 0, "physical": 0, "not_executable_only": 0, "beyond_policy_unprotected": 0}
print(f"retention_keep={keep}")
for s in snapshots:
    size = tree_bytes(snap_root / s.snapshot_id)
    unique = physical_bytes(snap_root / s.snapshot_id)
    totals["physical"] += unique
    states = sorted(by_snapshot.get(s.snapshot_id, []))
    prot = s.snapshot_id in protected
    ne_only = prot and all(st in ("NOT_EXECUTABLE", "COMPLETED") for st in states) and "NOT_EXECUTABLE" in states
    totals["all"] += size
    if ne_only:
        totals["not_executable_only"] += size
    if not prot and s.snapshot_id not in kept_by_policy:
        totals["beyond_policy_unprotected"] += size
    print(json.dumps({
        "snapshot_id": s.snapshot_id, "bytes": size, "new_physical_bytes": unique,
        "created_at": str(s.created_at),
        "retained_by_policy": s.snapshot_id in kept_by_policy, "protected_by_request": prot,
        "request_states": states, "protected_only_by_not_executable": ne_only,
    }, sort_keys=True))
for entry in sorted(snap_root.iterdir()) if snap_root.exists() else []:
    if entry.name not in listed:
        print(json.dumps({"unlisted_snapshot_entry": str(entry), "bytes": tree_bytes(entry) if entry.is_dir() else entry.stat().st_size}))
print("snapshot_totals " + json.dumps(totals, sort_keys=True))

part_re = re.compile(r"^part-(\d{9})-(\d{9})\.parquet$")
orphan_total = 0
for parts_dir in sorted(canonical.glob("*.parts")):
    parts = []
    for p in parts_dir.iterdir():
        m = part_re.match(p.name)
        if m and p.is_file():
            parts.append((int(m.group(1)), int(m.group(2)), p))
    for start, end, p in parts:
        if any((s, e) != (start, end) and s <= start and end <= e for s, e, _ in parts):
            size = p.stat().st_size
            orphan_total += size
            print(f"superseded_part bytes={size} path={p}")
    others = [p for p in parts_dir.iterdir() if not part_re.match(p.name)]
    for p in others:
        print(f"non_part_file_in_parts_dir bytes={p.stat().st_size if p.is_file() else -1} path={p}")
print(f"superseded_parts_total_bytes={orphan_total}")
PY
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
    inventory) op_inventory ;;
    checkpoint-select)
      runtime_python -m nflprops.platform.wizard_runtime checkpoint executable \
        || die "checkpoint select failed"
      ;;
    result-ingest)
      [ "$#" -eq 2 ] || die "result-ingest needs BUNDLE_ID MANIFEST_SHA256"
      [[ "$1" =~ ^checkpoint-result-[0-9a-f]{64}$ ]] || die "bundle_id format"
      [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die "manifest sha256 format"
      [ -d "$ROOT/publications/$1" ] || die "no published bundle $1"
      runtime_python -m nflprops.platform.wizard_runtime result-ingest \
        --bundle-id "$1" --expected-manifest-sha256 "$2" || die "result ingest failed"
      ;;
    outcome-ingest-hold)
      touch "$ROOT/state/outcome_ingest.hold" || die "could not create hold file"
      echo "HELD: $ROOT/state/outcome_ingest.hold"
      ;;
    outcome-ingest-release)
      rm -f "$ROOT/state/outcome_ingest.hold" || die "could not remove hold file"
      echo "RELEASED: recurring outcome ingest may run"
      ;;
    outcome-report)
      [ "$#" -le 1 ] || die "outcome-report takes at most one AS_OF"
      if [ "$#" -eq 1 ]; then
        [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\+00:00|Z)$ ]] || die "as_of format"
        runtime_python -m nflprops.platform.wizard_runtime outcome-report --as-of "$1" \
          || die "outcome report failed"
      else
        runtime_python -m nflprops.platform.wizard_runtime outcome-report || die "outcome report failed"
      fi
      echo "hold_file_present=$([ -e "$ROOT/state/outcome_ingest.hold" ] && echo yes || echo no)"
      ;;
    checkpoint-refuse)
      [ "$#" -eq 2 ] || die "checkpoint-refuse needs RUN_ID WORKFLOW_RUN_URL"
      [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "run_id format"
      [[ "$2" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+$ ]] || die "workflow run url format"
      runtime_python -m nflprops.platform.wizard_runtime checkpoint refuse \
        --run-id "$1" --workflow-run "$2" || die "checkpoint refuse failed"
      ;;
    ingest-stats)
      [ "$#" -ge 1 ] && [ "$#" -le 2 ] || die "ingest-stats needs SEASON [WEEKS]"
      [[ "$1" =~ ^20[0-9]{2}$ ]] || die "season format"
      if [ "$#" -eq 2 ]; then
        [[ "$2" =~ ^[0-9]{1,2}(,[0-9]{1,2})*$ ]] || die "weeks format"
        runtime_python -m nflprops.platform.wizard_runtime ingest-stats \
          --seasons "$1" --weeks "$2" || die "ingest-stats failed"
      else
        runtime_python -m nflprops.platform.wizard_runtime ingest-stats \
          --seasons "$1" || die "ingest-stats failed"
      fi
      ;;
    *) die "unknown operation $op" ;;
  esac
}

main "$@" </dev/null
