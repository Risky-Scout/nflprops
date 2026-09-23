#!/usr/bin/env bash
# BLOCK 2B: activate ONE already-prepared nflprops release on the Wizard
# host, gate it on the real runtime health, and roll back on failure.
# Streamed over SSH by .github/workflows/deploy-wizard.yml:
#
#   ssh ... bash -s -- RUNTIME_ROOT RELEASE_ID RELEASE_SHA \
#     < deploy/wizard/activate_release.sh
#
# Exit status (the workflow fails on anything but 0):
#   0  DEPLOYED     -- new release active and its health gate passed
#   1  ROLLED BACK  -- new release failed; previous release restored,
#                      restarted, and ITS health gate passed
#   2  ROLLBACK FAILED -- new release failed AND the restored previous
#                      release failed its health gate: HARD FAIL
#   3  NO ROLLBACK TARGET -- first deploy failed; `current` removed so
#                      nothing points at the broken release: HARD FAIL
#   4  refused before switching (release not prepared, bad arguments)
#
# The health gate for a release, after `current` points at it:
#   1. sudo -n systemctl restart nflprops-runtime.service
#   2. systemctl is-active nflprops-runtime.service
#   3. current/.venv/bin/python -m nflprops.cli platform health
#        --deploy-gate --expect-version <that release's SHA>
# All three must succeed; nothing here masks a failure.
#
# Runs as wizard-deploy. The only privileged action is the scoped
# `sudo -n systemctl restart nflprops-runtime.service` installed by the
# one-time root setup. Touches no other service, nginx, or any path
# outside RUNTIME_ROOT.
#
# Test hooks (never set by the workflow): NFLPROPS_BOOTSTRAP_PYTHON,
# NFLPROPS_SYSTEMCTL_CONTROL, NFLPROPS_SYSTEMCTL_QUERY.

set -euo pipefail

UNIT=nflprops-runtime.service

refuse() {
  echo "::error::ACTIVATION REFUSED: $*" >&2
  exit 4
}

main() {
  [ "$#" -eq 3 ] || refuse "usage: RUNTIME_ROOT RELEASE_ID RELEASE_SHA"
  RUNTIME_ROOT="$1"
  local release_id="$2" release_sha="$3"
  BOOTSTRAP_PYTHON="${NFLPROPS_BOOTSTRAP_PYTHON:-python3}"
  SYSTEMCTL_CONTROL="${NFLPROPS_SYSTEMCTL_CONTROL:-sudo -n /usr/bin/systemctl}"
  SYSTEMCTL_QUERY="${NFLPROPS_SYSTEMCTL_QUERY:-/usr/bin/systemctl}"
  CURRENT="$RUNTIME_ROOT/current"
  ENV_FILE="$RUNTIME_ROOT/nflprops-runtime.env"

  [[ "$release_sha" =~ ^[0-9a-f]{40}$ ]] || refuse "RELEASE_SHA must be a 40-char lowercase git SHA"
  [[ "$release_id" =~ ^${release_sha}-[0-9]+-[0-9]+$ ]] || refuse "RELEASE_ID must be <sha>-<run_id>-<attempt>"

  local new="$RUNTIME_ROOT/releases/$release_id"
  [ -f "$new/.prepared" ] || refuse "$new was not successfully prepared"
  [ -x "$new/.venv/bin/python" ] || refuse "$new/.venv/bin/python is missing"
  [ "$(cat "$new/RELEASE_SHA")" = "$release_sha" ] || refuse "$new/RELEASE_SHA does not match $release_sha"
  [ -f "$ENV_FILE" ] || refuse "$ENV_FILE is missing"

  local previous=""
  if [ -L "$CURRENT" ]; then previous="$(readlink "$CURRENT")"; fi
  [ "$previous" != "$new" ] || refuse "$new is already current"

  if [ -n "$previous" ] && [ ! -f "$previous/RELEASE_SHA" ]; then
    refuse "previous release $previous has no RELEASE_SHA -- it could not be health-verified on rollback"
  fi

  echo "ACTIVATING: $new (previous: ${previous:-<none>})"
  switch_current "$new" || refuse "could not switch current to $new (current unchanged)"

  if health_gate "$release_sha"; then
    prune_releases "$new" "$previous"
    echo "DEPLOYED: $new"
    return 0
  fi

  echo "::error::new release $release_id FAILED its runtime health gate -- rolling back" >&2
  if [ -z "$previous" ]; then
    rm -f "$CURRENT"
    echo "::error::HARD FAIL: no previous release to roll back to; current removed so nothing points at the broken release" >&2
    return 3
  fi

  if ! switch_current "$previous"; then
    echo "::error::HARD FAIL: ROLLBACK FAILED -- could not restore current to $previous" >&2
    return 2
  fi
  local previous_sha
  previous_sha="$(cat "$previous/RELEASE_SHA")"
  if health_gate "$previous_sha"; then
    echo "::error::ROLLED BACK to $previous (health verified); deployment of $release_id FAILED" >&2
    return 1
  fi
  echo "::error::HARD FAIL: ROLLBACK FAILED -- restored $previous but it did not pass its runtime health gate. Manual intervention required." >&2
  return 2
}

# Atomic: build the new symlink beside `current`, then rename(2) it over
# `current` -- readers only ever see the old or the new target.
switch_current() {
  ln -sfn "$1" "$CURRENT.new" || return 1
  "$BOOTSTRAP_PYTHON" -c 'import os, sys; os.replace(sys.argv[1], sys.argv[2])' "$CURRENT.new" "$CURRENT"
}

health_gate() {
  local expect_sha="$1"
  echo "HEALTH GATE: restart $UNIT (expect $expect_sha)"
  if ! $SYSTEMCTL_CONTROL restart "$UNIT"; then
    echo "HEALTH GATE: restart $UNIT failed" >&2
    return 1
  fi
  if ! $SYSTEMCTL_QUERY is-active --quiet "$UNIT"; then
    echo "HEALTH GATE: $UNIT is not active" >&2
    return 1
  fi
  if ! (
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    "$CURRENT/.venv/bin/python" -m nflprops.cli platform health \
      --deploy-gate --expect-version "$expect_sha"
  ); then
    echo "HEALTH GATE: platform health --deploy-gate failed" >&2
    return 1
  fi
  echo "HEALTH GATE: passed ($expect_sha)"
}

# Bounded disk: keep exactly the active release and its rollback target.
prune_releases() {
  local keep_new="$1" keep_previous="$2" entry
  for entry in "$RUNTIME_ROOT"/releases/*; do
    [ -e "$entry" ] || continue
    [ "$entry" = "$keep_new" ] && continue
    [ -n "$keep_previous" ] && [ "$entry" = "$keep_previous" ] && continue
    rm -rf "$entry"
  done
}

main "$@" </dev/null
