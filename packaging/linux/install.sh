#!/bin/sh
# Install Vantage for the current user (no root needed):
#   ~/.local/opt/vantage            the program
#   ~/.local/bin/vantage            command for batch mode, --self-test and starting the window from a terminal
#   menu entry, icon and the .vantage file type
# Run it from the extracted folder:  ./install.sh      (PREFIX=/some/where ./install.sh to install elsewhere)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${PREFIX:-$HOME/.local}
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
mkdir -p "$DEST" "$PREFIX/bin" "$PREFIX/share/applications" "$PREFIX/share/icons/hicolor/512x512/apps" \
         "$PREFIX/share/mime/packages"
cp -R "$HERE/." "$DEST/"
touch "$DEST/$MARK"
ln -sf "$DEST/vantage-cli" "$PREFIX/bin/vantage"
cp "$DEST/_internal/vantage/resources/icons/vantage.png" "$PREFIX/share/icons/hicolor/512x512/apps/vantage.png"
sed "s|@EXEC@|\"$DEST/Vantage\"|" "$HERE/vantage.desktop" > "$PREFIX/share/applications/vantage.desktop"
cp "$HERE/vantage-mime.xml" "$PREFIX/share/mime/packages/vantage.xml"

command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$PREFIX/share/mime" >/dev/null 2>&1 || true
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$PREFIX/share/applications" >/dev/null 2>&1 || true

echo "Vantage installed in $DEST"
echo "  menu:     Vantage (under Engineering / Graphics)"
echo "  terminal: $PREFIX/bin/vantage   (vantage --self-test checks the installation)"
case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "  note: $PREFIX/bin is not on your PATH" ;;
esac
