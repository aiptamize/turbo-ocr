#!/bin/bash
# Install fastpdf2png v2.0 (PDF renderer for the OCR server).
# Clones, builds, and copies the binary + library to bin/.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$ROOT/bin"
REPO="https://github.com/nataell95/fastpdf2png.git"
BRANCH="main"
TMP_DIR="/tmp/fastpdf2png_build_$$"

echo "=== Installing fastpdf2png ==="

# Skip if already installed and recent
if [ -x "$BIN_DIR/fastpdf2png" ] && [ -f "$BIN_DIR/libpdfium.so" ] && [ -f "$BIN_DIR/libfastpdf2png.so" ]; then
  echo "fastpdf2png already installed in $BIN_DIR"
  echo "  Delete bin/fastpdf2png to force reinstall."
  exit 0
fi

mkdir -p "$BIN_DIR"

# Clone
echo "Cloning $REPO ($BRANCH)..."
git clone --depth 1 --branch "$BRANCH" "$REPO" "$TMP_DIR"

# Build
echo "Building..."
cd "$TMP_DIR"
bash scripts/build.sh

# Copy artifacts
cp build/fastpdf2png "$BIN_DIR/"
cp build/libpdfium.so "$BIN_DIR/" 2>/dev/null || cp pdfium/lib/libpdfium.so "$BIN_DIR/"
# Copy shared library (v2.0+ splits core into libfastpdf2png.so)
if ls build/libfastpdf2png.so* 1>/dev/null 2>&1; then
  cp -a build/libfastpdf2png.so* "$BIN_DIR/"
fi
chmod +x "$BIN_DIR/fastpdf2png"

# Cleanup
rm -rf "$TMP_DIR"

echo ""
echo "=== Installed ==="
echo "  Binary:  $BIN_DIR/fastpdf2png"
echo "  Library: $BIN_DIR/libpdfium.so"
echo "  Library: $BIN_DIR/libfastpdf2png.so"
echo ""
echo "  Verify: $BIN_DIR/fastpdf2png --info some.pdf"
