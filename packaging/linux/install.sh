#!/bin/sh
# Install Vantage for the current user (no root needed):
#   ~/.local/opt/vantage            the program
#   ~/.local/bin/vantage            command for batch mode, --self-test and starting the window from a terminal
#   menu entry, icon and the .vantage file type in $XDG_DATA_HOME (default ~/.local/share)
# Run it from the extracted folder:  ./install.sh
# PREFIX=/some/where ./install.sh installs into /some/where/opt/vantage, /some/where/bin and /some/where/share instead.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${PREFIX:-}" ]; then
    DATA="$PREFIX/share"
else
    PREFIX="$HOME/.local"
    DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
fi
DEST="$PREFIX/opt/vantage"
MARK=.vantage-install

if [ ! -x "$HERE/Vantage" ] || [ ! -x "$HERE/vantage-cli" ]; then
    echo "install.sh: run me from the extracted Vantage folder (Vantage and vantage-cli must be next to me)" >&2
    exit 1
fi
if [ -e "$DEST" ] && [ ! -e "$DEST/$MARK" ]; then
    echo "install.sh: $DEST exists and was not made by this installer; move it away first" >&2
    exit 1
fi
if [ "$HERE" = "$DEST" ]; then
    echo "install.sh: already installed in $DEST" >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST" "$PREFIX/bin" "$DATA/applications" "$DATA/icons/hicolor/512x512/apps" "$DATA/mime/packages"
cp -R "$HERE/." "$DEST/"
printf 'data=%s\nbin=%s\n' "$DATA" "$PREFIX/bin" > "$DEST/$MARK"     # where uninstall.sh finds what this wrote
ln -sf "$DEST/vantage-cli" "$PREFIX/bin/vantage"
cp "$DEST/_internal/vantage/resources/icons/vantage.png" "$DATA/icons/hicolor/512x512/apps/vantage.png"

# the Exec line, as the Desktop Entry spec wants it: the path in double quotes with " ` $ \ escaped by a backslash,
# that backslash (and a literal one) escaped again because a desktop file's values are unescaped before the quotes
# are read (GLib rejects the entry otherwise), and % doubled
exe=$(printf '%s' "$DEST/Vantage" | sed -e 's/\\/\\\\\\\\/g' -e 's/["`$]/\\\\&/g' -e 's/%/%%/g')
while IFS= read -r line; do
    case "$line" in
        Exec=*) printf 'Exec="%s" %%f\n' "$exe" ;;
        *) printf '%s\n' "$line" ;;
    esac
done < "$HERE/vantage.desktop" > "$DATA/applications/vantage.desktop"
cp "$HERE/vantage-mime.xml" "$DATA/mime/packages/vantage.xml"

command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$DATA/mime" >/dev/null 2>&1 || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true

echo "Vantage installed in $DEST"
echo "  menu:      Vantage (under Science / Engineering)"
echo "  terminal:  $PREFIX/bin/vantage   (vantage --self-test checks the installation)"
echo "  uninstall: $DEST/uninstall.sh"
case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "  note: $PREFIX/bin is not on your PATH" ;;
esac
