#!/usr/bin/env bash
# deploy/install.sh – idempotent install script for EDGE-AURA-MFP on a Pi 3B+.
#
# What this script does (in order):
#
#   1. Creates the unprivileged ``aura-edge`` system user.
#   2. Creates /opt/aura-edge and owns it to aura-edge.
#   3. Creates a Python venv at /opt/aura-edge/venv and installs this
#      package in editable mode (``pip install -e .``).
#   4. Copies the systemd unit files to /etc/systemd/system/ and installs
#      a tiny /opt/aura-edge/bin/run-bridge wrapper.
#   5. Runs ``systemctl daemon-reload`` and enables the two services so
#      they come up on next boot.
#   6. Does *not* start the services – you do that by hand once the Pico
#      is wired up: ``sudo systemctl start aura-edge-daemon aura-edge-bridge``.
#
# Re-runs are safe: every step checks for "already there" before acting.
#
# Usage:
#     sudo deploy/install.sh               # normal install
#     deploy/install.sh --dry-run          # print every action, do nothing
#
# --dry-run DOES NOT require root and exits 0, which makes it usable from
# CI (validation gate 5).

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --help|-h)
            sed -n '2,24p' "$0"
            exit 0
            ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_ROOT="/opt/aura-edge"
VENV_DIR="${INSTALL_ROOT}/venv"
BIN_DIR="${INSTALL_ROOT}/bin"
USER_NAME="aura-edge"
SYSTEMD_DIR="/etc/systemd/system"
LOG_DIR="${INSTALL_ROOT}/logs"
ENV_DIR="/etc/aura-edge"

# ─────────────────────────────────────────────────────────────────────
# Helpers – every mutating action goes through ``run`` so --dry-run
# turns it into an informational ``echo``.
# ─────────────────────────────────────────────────────────────────────
run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] %q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

info() { printf '[install] %s\n' "$*"; }

require_root() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then return 0; fi
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "install.sh: must run as root (or pass --dry-run)" >&2
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────
# 1. Create the aura-edge user
# ─────────────────────────────────────────────────────────────────────
ensure_user() {
    info "ensuring user '${USER_NAME}' exists"
    if [[ "${DRY_RUN}" -eq 0 ]] && id -u "${USER_NAME}" >/dev/null 2>&1; then
        info "user ${USER_NAME} already exists – skipping"
        return
    fi
    run useradd --system --home-dir "${INSTALL_ROOT}" --shell /usr/sbin/nologin "${USER_NAME}" || true
    # Serial port access on the Pi lives in the dialout group.
    run usermod -a -G dialout "${USER_NAME}" || true
}

# ─────────────────────────────────────────────────────────────────────
# 2. Directory layout
# ─────────────────────────────────────────────────────────────────────
ensure_dirs() {
    info "ensuring directory layout under ${INSTALL_ROOT}"
    for d in "${INSTALL_ROOT}" "${BIN_DIR}" "${LOG_DIR}" "${ENV_DIR}"; do
        run mkdir -p "${d}"
    done
    run chown -R "${USER_NAME}:${USER_NAME}" "${INSTALL_ROOT}"
}

# ─────────────────────────────────────────────────────────────────────
# 3. Venv + pip install -e .
# ─────────────────────────────────────────────────────────────────────
ensure_venv() {
    info "ensuring venv at ${VENV_DIR}"
    if [[ "${DRY_RUN}" -eq 0 ]] && [[ -x "${VENV_DIR}/bin/python" ]]; then
        info "venv already present – upgrading package in place"
    else
        run python3 -m venv "${VENV_DIR}"
    fi
    run "${VENV_DIR}/bin/pip" install --upgrade pip
    run "${VENV_DIR}/bin/pip" install -e "${REPO_ROOT}"
    run chown -R "${USER_NAME}:${USER_NAME}" "${VENV_DIR}"
}

# ─────────────────────────────────────────────────────────────────────
# 4. Install systemd units + run-bridge wrapper
# ─────────────────────────────────────────────────────────────────────
install_units() {
    info "installing systemd unit files into ${SYSTEMD_DIR}"
    for unit in aura-edge-daemon.service aura-edge-bridge.service; do
        run install -m 0644 "${REPO_ROOT}/deploy/systemd/${unit}" "${SYSTEMD_DIR}/${unit}"
    done

    info "installing bridge wrapper ${BIN_DIR}/run-bridge"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] write %s\n' "${BIN_DIR}/run-bridge"
    else
        cat >"${BIN_DIR}/run-bridge" <<WRAP
#!/usr/bin/env bash
# Pipe the daemon's journald output into the orchestrator bridge.
set -euo pipefail
exec journalctl -f -u aura-edge-daemon.service -o cat --output-fields=MESSAGE \\
  | ${VENV_DIR}/bin/python -m pi.orchestrator_bridge
WRAP
    fi
    run chmod 0755 "${BIN_DIR}/run-bridge"

    # Seed an empty env file so EnvironmentFile=-... has something to read.
    if [[ "${DRY_RUN}" -eq 0 ]] && [[ ! -f "${ENV_DIR}/bridge.env" ]]; then
        run touch "${ENV_DIR}/bridge.env"
        run chmod 0644 "${ENV_DIR}/bridge.env"
    elif [[ "${DRY_RUN}" -eq 1 ]]; then
        run touch "${ENV_DIR}/bridge.env"
    fi

    # Seed a *template* station.env (commented out) — the daemon service
    # requires AURA_STATION_LAT/AURA_STATION_LON (no `-` prefix on its
    # EnvironmentFile=, so it fails to start rather than guessing a
    # location). We deliberately do NOT invent coordinates here: an
    # installer must fill these in from an actual site survey before
    # starting aura-edge-daemon.
    if [[ "${DRY_RUN}" -eq 0 ]] && [[ ! -f "${ENV_DIR}/station.env" ]]; then
        cat > "${ENV_DIR}/station.env" <<'STATIONENV'
# Fixed-install station coordinates for pi.daemon (no GPS on the BOM).
# Uncomment and set both before starting aura-edge-daemon.service:
# AURA_STATION_LAT=36.17
# AURA_STATION_LON=-86.78
STATIONENV
        run chmod 0644 "${ENV_DIR}/station.env"
        info "wrote template ${ENV_DIR}/station.env — edit it with real site coordinates before starting aura-edge-daemon"
    elif [[ "${DRY_RUN}" -eq 1 ]]; then
        run touch "${ENV_DIR}/station.env"
    fi
}

# ─────────────────────────────────────────────────────────────────────
# 5. Reload + enable (but don't start)
# ─────────────────────────────────────────────────────────────────────
enable_units() {
    info "reloading systemd + enabling units"
    run systemctl daemon-reload
    run systemctl enable aura-edge-daemon aura-edge-bridge
    info "services are ENABLED but NOT STARTED – run 'sudo systemctl start aura-edge-daemon aura-edge-bridge' once the hardware is connected"
}

main() {
    require_root
    ensure_user
    ensure_dirs
    ensure_venv
    install_units
    enable_units
    info "done."
}

main "$@"
