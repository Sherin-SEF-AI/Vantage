#!/bin/sh
# Remove a per-user Vantage installation made by install.sh. Projects and settings (~/.config/Vantage) are kept.
set -eu
PREFIX=${PREFIX:-$HOME/.local}
DEST="$PREFIX/opt/vantage"

if [ -e "$DEST" ] && [ ! -e "$DEST/.vantage-install" ]; then
    echo "uninstall.sh: $DEST was not made by install.sh; not touching it" >&2
    exit 1
fi
link="$PREFIX/bin/vantage"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$DEST/vantage-cli" ]; then
    rm -f "$link"
fi
rm -f "$PREFIX/share/applications/vantage.desktop" "$PREFIX/share/icons/hicolor/512x512/apps/vantage.png" \
      "$PREFIX/share/mime/packages/vantage.xml"
rm -rf "$DEST"
command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$PREFIX/share/mime" >/dev/null 2>&1 || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$PREFIX/share/applications" >/dev/null 2>&1 || true
echo "Vantage removed (your projects and ~/.config/Vantage are kept)"
