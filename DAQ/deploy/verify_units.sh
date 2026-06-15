#!/usr/bin/env bash
# deploy/verify_units.sh – CI-friendly wrapper for ``systemd-analyze verify``.
#
# ``systemd-analyze verify`` parses each unit file and runs a number of
# sanity checks.  On a build machine the binaries referenced by
# ``ExecStart=`` (``/opt/aura-edge/venv/bin/python``) don't exist yet,
# which makes the tool exit 1 with "Command ... is not executable".
# That doesn't mean the unit file is malformed – it just means the
# install hasn't happened yet.  This wrapper treats only that specific
# error as benign and re-raises everything else (syntax errors,
# unknown directives, circular After= chains, etc.).
#
# If ``systemd-analyze`` is unavailable, falls back to a small shell
# lint that checks for required sections and ``ExecStart=``.

set -euo pipefail

units=("$@")
if [[ ${#units[@]} -eq 0 ]]; then
    echo "usage: $0 <unit-file> [unit-file ...]" >&2
    exit 2
fi

if command -v systemd-analyze >/dev/null 2>&1; then
    # Capture stdout+stderr.  Tolerate only "is not executable" errors.
    output=$(systemd-analyze verify "${units[@]}" 2>&1 || true)
    if [[ -n "${output}" ]]; then
        # Strip the benign "Command ... is not executable" lines.
        residual=$(printf '%s\n' "${output}" | grep -vE 'is not executable: No such file or directory' || true)
        if [[ -n "${residual}" ]]; then
            echo "systemd-analyze reported real issues:" >&2
            echo "${residual}" >&2
            exit 1
        fi
        echo "systemd-analyze: only 'missing ExecStart binary' warnings (benign on a build host)"
    fi
    echo "systemd units parse cleanly:"
    for u in "${units[@]}"; do echo "  OK  $u"; done
    exit 0
fi

# Fallback: minimal shell lint.
echo "systemd-analyze not available – falling back to shell lint"
rc=0
for u in "${units[@]}"; do
    for required in '^\[Unit\]' '^\[Service\]' '^\[Install\]' '^ExecStart='; do
        if ! grep -qE "${required}" "${u}"; then
            echo "  FAIL ${u}: missing ${required}" >&2
            rc=1
        fi
    done
    if [[ ${rc} -eq 0 ]]; then echo "  OK  ${u}"; fi
done
exit "${rc}"
