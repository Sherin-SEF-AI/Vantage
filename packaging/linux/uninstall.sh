#!/bin/sh
# Remove a per-user Vantage installation made by install.sh (run with the same PREFIX if you gave one). Projects and
# settings (~/.config/Vantage) are kept; a menu entry or file type that `vantage --install-desktop` wrote is left alone.
set -eu
if [ -n "${PREFIX:-}" ]; then
    DATA="$PREFIX/share"
else
    PREFIX="$HOME/.local"
    DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
fi
DEST="$PREFIX/opt/vantage"
MARK="$DEST/.vantage-install"
BIN="$PREFIX/bin"

if [ -e "$DEST" ] && [ ! -e "$MARK" ]; then
    echo "uninstall.sh: $DEST was not made by install.sh; not touching it" >&2
    exit 1
fi
if [ -e "$MARK" ]; then                  # where install.sh put the menu entry and the command
    d=$(sed -n 's/^data=//p' "$MARK")
    b=$(sed -n 's/^bin=//p' "$MARK")
    DATA=${d:-$DATA}
    BIN=${b:-$BIN}
fi
link="$BIN/vantage"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$DEST/vantage-cli" ]; then
    rm -f "$link"
fi
entry="$DATA/applications/vantage.desktop"
exe=$(printf '%s' "$DEST/Vantage" | sed -e 's/\\/\\\\\\\\/g' -e 's/["`$]/\\\\&/g' -e 's/%/%%/g')   # as install.sh wrote it
if [ -f "$entry" ] && grep -qxF "Exec=\"$exe\" %f" "$entry"; then
    rm -f "$entry"
fi
mime="$DATA/mime/packages/vantage.xml"
if [ -f "$mime" ] && ! grep -q 'written by vantage --install-desktop' "$mime"; then
    rm -f "$mime"
fi
if [ ! -e "$entry" ]; then               # the icon goes with the entry that uses it
    rm -f "$DATA/icons/hicolor/512x512/apps/vantage.png"
fi
rm -rf "$DEST"
command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$DATA/mime" >/dev/null 2>&1 || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
echo "Vantage removed (your projects and ~/.config/Vantage are kept)"
