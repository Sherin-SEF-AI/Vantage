#!/usr/bin/env bash
# Install Vantage from this checkout on an NVIDIA Jetson AGX Orin (JetPack 7: L4T R39, Ubuntu 24.04, CUDA 13).
#
#   packaging/jetson/setup.sh [--no-torch] [--no-desktop] [--self-test]
#
# Makes .venv/ in the checkout with access to the system's packages (apt's OpenCV with GStreamer, numpy, scipy),
# installs the rest from PyPI, PyTorch with CUDA from PyTorch's cu130 index (the aarch64 wheels run on Orin's GPU:
# the self-test's gpu line checks the results against the CPU), and Vantage itself in editable mode. Then
# ~/.local/bin/vantage, the app-menu entry and the .vantage file type (vantage --install-desktop). Run it again to
# update; it changes nothing outside the checkout, ~/.local/bin/vantage and the desktop entries.
set -euo pipefail

TORCH=1 DESKTOP=1 SELFTEST=0
for a in "$@"; do
    case "$a" in
        --no-torch) TORCH=0 ;;
        --no-desktop) DESKTOP=0 ;;
        --self-test) SELFTEST=1 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Checking the system"
[ "$(uname -m)" = aarch64 ] || echo "warning: $(uname -m), not aarch64: this script is meant for a Jetson" >&2
if [ -r /etc/nv_tegra_release ]; then head -1 /etc/nv_tegra_release; else echo "warning: no /etc/nv_tegra_release (not L4T?)" >&2; fi
missing=()
/usr/bin/python3 -c 'import ensurepip, venv' 2>/dev/null || missing+=(python3-venv)
for pkg in python3-opencv python3-numpy python3-scipy gstreamer1.0-tools; do
    dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed' || missing+=("$pkg")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing system packages. Install them, then run this again:" >&2
    echo "  sudo apt install ${missing[*]}" >&2
    exit 1
fi
/usr/bin/python3 -c 'import cv2, sys; sys.exit(0 if "GStreamer:                   YES" in cv2.getBuildInformation() else 1)' \
    || echo "warning: the system's OpenCV has no GStreamer; pipelines will run through gst-launch-1.0 instead" >&2

say "Virtual environment: $VENV"
[ -x "$PY" ] || /usr/bin/python3 -m venv --system-site-packages "$VENV"
"$PY" -m pip install -q --upgrade pip
# PyQt6 inside the venv (not a copy in ~/.local, which the venv would otherwise also see)
"$PY" -m pip install -q --ignore-installed 'PyQt6>=6.5'
"$PY" -m pip install -q -r "$ROOT/packaging/jetson/requirements-jetson.txt"
# numpy and scipy from PyPI inside the venv: their wheels bring OpenBLAS, while apt's use whatever libblas.so.3 points
# at, on a fresh JetPack the reference BLAS (a 2000 x 2000 matmul in 5.5 s instead of ~0.1 s on an AGX Orin). numpy
# stays below 2: apt's OpenCV is built against numpy 1.x.
"$PY" -m pip install -q --ignore-installed 'numpy>=1.24,<2' 'scipy>=1.10,<1.17'

if [ "$TORCH" = 1 ]; then
    say "PyTorch with CUDA 13 (cu130)"
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu130
fi

say "Vantage (editable, without PyPI's OpenCV)"
"$PY" -m pip install -q --no-deps -e "$ROOT"
# Guard: a PyPI OpenCV (pulled in by something else) would hide apt's GStreamer build
if "$PY" -m pip show -q opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1; then
    echo "removing PyPI OpenCV from the venv (it has no GStreamer)"
    "$PY" -m pip uninstall -y -q opencv-python opencv-python-headless opencv-contrib-python || true
fi

mkdir -p "$HOME/.local/bin"
if [ ! -e "$HOME/.local/bin/vantage" ] || [ -L "$HOME/.local/bin/vantage" ]; then
    ln -sfn "$VENV/bin/vantage" "$HOME/.local/bin/vantage"
    echo "~/.local/bin/vantage -> $VENV/bin/vantage"
else
    echo "~/.local/bin/vantage exists and is not a link: left alone (run $VENV/bin/vantage)"
fi
[ "$DESKTOP" = 1 ] && "$VENV/bin/vantage" --install-desktop || true

say "Summary"
"$PY" - <<'EOF'
import cv2, numpy, PyQt6.QtCore as Q
gst = 'GStreamer:                   YES' in cv2.getBuildInformation()
print(f'OpenCV {cv2.__version__} ({"with" if gst else "without"} GStreamer) · numpy {numpy.__version__} · PyQt6 {Q.PYQT_VERSION_STR}')
try:
    from vantage.selftest import _check_gpu
    print('gpu:', _check_gpu({})[1])
except Exception as e:
    print('gpu:', e)
EOF
if [ "$SELFTEST" = 1 ]; then
    say "Self-test (about 5 minutes)"
    QT_QPA_PLATFORM=offscreen "$VENV/bin/vantage" --self-test
else
    echo; echo "Done. Start with: vantage --sample      Check everything: vantage --self-test"
fi
