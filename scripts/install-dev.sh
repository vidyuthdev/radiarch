#!/usr/bin/env bash
# install-dev.sh — Reproducible editable install for Radiarch.
#
# Purges the stale build artifacts that cause modern setuptools'
# *strict* editable install to ship a finder with out-of-date
# NAMESPACES (which breaks vendored OpenTPS submodule imports such as
# `opentps.core.processing.doseCalculation.protons.MCsquare`).
#
# Use this any time someone adds a new package directory under
# `src/opentps/` or `src/radiarch/`, or any time `pytest` blows up
# with `ModuleNotFoundError: No module named 'opentps.core.processing'`.
#
# Usage:
#   ./scripts/install-dev.sh            # editable install + dev extras
#   ./scripts/install-dev.sh --no-deps  # skip resolving runtime deps

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"

echo "==> Cleaning stale build artifacts in ${SRC_DIR}"
rm -rf \
  "${SRC_DIR}"/*.egg-info \
  "${SRC_DIR}"/build \
  "${SRC_DIR}"/dist

# Find every __editable__*radiarch* finder/pth in the active venv
# and remove it so the next install regenerates it from scratch.
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "==> Removing stale editable hooks under ${SITE_PACKAGES}"
find "${SITE_PACKAGES}" -maxdepth 1 \
  \( -name "__editable__*radiarch*" -o -name "radiarch-*.dist-info" \) \
  -exec rm -rf {} + 2>/dev/null || true

echo "==> pip uninstall radiarch (ignore-not-found)"
pip uninstall -y radiarch 2>/dev/null || true

# Filter out anything that looks like a leftover shell comment —
# guards against users pasting commands from markdown into a zsh
# that doesn't have `interactive_comments` set, where `#` is passed
# as a literal argument and pip blows up on "Invalid requirement: '#'".
# Use a safe-with-set-u pattern: only iterate if $# > 0.
ARGS=()
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      "#"|"#"*) ;;  # skip
      *) ARGS+=("$arg") ;;
    esac
  done
fi

# Echo with safe expansion (set -u trips on ${ARGS[*]} when empty under bash<4.4).
echo "==> pip install -e ${SRC_DIR} ${ARGS[*]:-}"
if [ "${#ARGS[@]}" -gt 0 ]; then
  pip install -e "${SRC_DIR}" "${ARGS[@]}"
else
  pip install -e "${SRC_DIR}"
fi

echo
echo "==> Sanity check: import opentps.core.processing.doseCalculation.protons.MCsquare"
python3 -c "
import opentps.core.processing.doseCalculation.protons.MCsquare as M
print('   OK ->', M.__path__[0])
"
echo
echo "Done. You can now run: pytest tests/"
