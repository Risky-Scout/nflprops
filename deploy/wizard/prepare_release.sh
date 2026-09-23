#!/usr/bin/env bash
# BLOCK 2B: prepare ONE immutable nflprops release on the Wizard host,
# entirely BEFORE activation. Streamed over SSH by
# .github/workflows/deploy-wizard.yml:
#
#   ssh ... bash -s -- RUNTIME_ROOT RELEASE_ID RELEASE_SHA UNIT_SHA256 \
#     < deploy/wizard/prepare_release.sh
#
# Expects the release archive already uploaded to
# RUNTIME_ROOT/releases/RELEASE_ID.tar.gz. On success the release
# RUNTIME_ROOT/releases/RELEASE_ID/ contains the source tree, its OWN
# virtualenv (.venv, `.[orchestration,runtime]` only -- no training-only
# extras), a RELEASE_SHA marker, and finally a .prepared marker that
# activate_release.sh requires. On ANY failure the partial release is
# deleted and RUNTIME_ROOT/current is never touched (it is only ever
# changed by activate_release.sh).
#
# Runs as wizard-deploy, no root. Touches nothing outside RUNTIME_ROOT.
# Never runs training, replay, simulation, or calibration -- only imports,
# --help, a read-only snapshot listing, and the read-only health gate.
#
# Test hooks (never set by the workflow): NFLPROPS_BOOTSTRAP_PYTHON,
# NFLPROPS_UNIT_DIR, NFLPROPS_CONTROL_PREFLIGHT.

set -euo pipefail

UNIT=nflprops-runtime.service

fail() {
  echo "::error::PREPARE FAILED: $*" >&2
  exit 1
}

main() {
  [ "$#" -eq 4 ] || fail "usage: RUNTIME_ROOT RELEASE_ID RELEASE_SHA UNIT_SHA256"
  local runtime_root="$1" release_id="$2" release_sha="$3" unit_sha256="$4"
  local bootstrap_python="${NFLPROPS_BOOTSTRAP_PYTHON:-python3}"
  local unit_dir="${NFLPROPS_UNIT_DIR:-/etc/systemd/system}"
  local control_preflight="${NFLPROPS_CONTROL_PREFLIGHT:-sudo -n -l /usr/bin/systemctl restart $UNIT}"

  case "$runtime_root" in
    /*) ;;
    *) fail "RUNTIME_ROOT must be absolute, got $runtime_root" ;;
  esac
  case "$runtime_root" in
    /home/wizard-deploy/nfl-production-2026* | /var/www/sportsodds*)
      fail "RUNTIME_ROOT $runtime_root overlaps an unrelated workload" ;;
  esac
  [[ "$release_sha" =~ ^[0-9a-f]{40}$ ]] || fail "RELEASE_SHA must be a 40-char lowercase git SHA"
  [[ "$release_id" =~ ^${release_sha}-[0-9]+-[0-9]+$ ]] || fail "RELEASE_ID must be <sha>-<run_id>-<attempt>"
  [[ "$unit_sha256" =~ ^[0-9a-f]{64}$ ]] || fail "UNIT_SHA256 must be a 64-char hex SHA-256"

  local releases="$runtime_root/releases"
  local release_dir="$releases/$release_id"
  local archive="$releases/$release_id.tar.gz"
  local current="$runtime_root/current"
  local env_file="$runtime_root/nflprops-runtime.env"

  local current_before=""
  if [ -L "$current" ]; then current_before="$(readlink "$current")"; fi

  [ -f "$archive" ] || fail "release archive $archive was not uploaded"
  [ ! -e "$release_dir" ] || fail "$release_dir already exists -- releases are immutable"

  # From here on, any failure deletes the partial release (never current).
  PREPARE_RELEASE_DIR="$release_dir"
  PREPARE_ARCHIVE="$archive"
  trap cleanup_on_failure EXIT

  # ---- preflight: fail before building anything ----------------------
  "$bootstrap_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "$bootstrap_python is older than Python 3.11"
  "$bootstrap_python" -m venv --help >/dev/null || fail "$bootstrap_python has no venv module"

  local installed_unit="$unit_dir/$UNIT"
  [ -f "$installed_unit" ] || fail "$installed_unit is not installed -- run the one-time root setup (docs/PLATFORM_AUTOMATION.md)"
  local installed_sha
  installed_sha="$("$bootstrap_python" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$installed_unit")"
  [ "$installed_sha" = "$unit_sha256" ] \
    || fail "installed $UNIT differs from this release's deploy/systemd/$UNIT -- re-run the one-time root install"
  $control_preflight >/dev/null \
    || fail "wizard-deploy may not 'sudo -n systemctl restart $UNIT' -- install the scoped sudoers rule (one-time root setup)"

  local free_kb
  free_kb="$(df -Pk "$runtime_root" | awk 'NR==2 {print $4}')"
  [ "$free_kb" -ge $((3 * 1024 * 1024)) ] || fail "less than 3 GiB free under $runtime_root"

  # ---- layout (as wizard-deploy, no root) ----------------------------
  local d
  for d in releases state state/warehouse snapshots publications backups logs locks; do
    install -d -m 750 "$runtime_root/$d"
  done

  # ---- build the immutable release -----------------------------------
  mkdir -m 750 "$release_dir"
  tar -xzf "$archive" -C "$release_dir"
  rm -f "$archive"
  printf '%s\n' "$release_sha" > "$release_dir/RELEASE_SHA"

  "$bootstrap_python" -m venv "$release_dir/.venv"
  local vpy="$release_dir/.venv/bin/python"
  [ -x "$vpy" ] || fail "$vpy was not created"
  "$vpy" -m pip install --no-cache-dir --disable-pip-version-check --quiet \
    -e "${release_dir}[orchestration,runtime]" \
    || fail "dependency installation failed"

  # The env file is created ONLY if absent -- never overwritten.
  if [ ! -f "$env_file" ]; then
    install -m 600 "$release_dir/deploy/systemd/nflprops-runtime.env.example" "$env_file"
  fi

  # ---- pre-activation checks, all from the CANDIDATE release's venv --
  (
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
    "$vpy" -c 'import nflprops, nflprops.cli, nflprops.platform.health, nflprops.platform.wizard_runtime' \
      || fail "nflprops does not import"
    "$vpy" -m nflprops.cli platform health --help >/dev/null || fail "platform health CLI does not run"
    "$vpy" -m nflprops.platform.wizard_runtime --help >/dev/null || fail "runtime entrypoint does not run"
    "$vpy" -m nflprops.platform.wizard_runtime snapshot --help >/dev/null || fail "snapshot CLI does not run"
    "$vpy" -m nflprops.platform.wizard_runtime snapshot list >/dev/null || fail "snapshot list (read-only) failed"
    "$vpy" -m nflprops.cli platform health --deploy-gate --expect-version "$release_sha" \
      || fail "candidate release failed the deployment-critical health gate"
  )

  local current_after=""
  if [ -L "$current" ]; then current_after="$(readlink "$current")"; fi
  [ "$current_after" = "$current_before" ] || fail "current changed during preparation"

  touch "$release_dir/.prepared"
  PREPARE_RELEASE_DIR=""
  PREPARE_ARCHIVE=""
  echo "PREPARED: $release_dir (current untouched: ${current_before:-<none>})"
}

cleanup_on_failure() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    [ -z "${PREPARE_RELEASE_DIR:-}" ] || rm -rf "$PREPARE_RELEASE_DIR"
    [ -z "${PREPARE_ARCHIVE:-}" ] || rm -f "$PREPARE_ARCHIVE"
    echo "PREPARE FAILED: partial release removed; current untouched" >&2
  fi
  exit "$status"
}

main "$@" </dev/null
