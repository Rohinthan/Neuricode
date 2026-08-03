#!/bin/bash
# download.sh — Download MNIST dataset with multiple mirror fallbacks

set -e
cd "$(dirname "$0")"

echo "╔══════════════════════════════════╗"
echo "║   MNIST Dataset Downloader       ║"
echo "╚══════════════════════════════════╝"

# ── mirrors (tried in order) ───────────────────────────────────────
MIRRORS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
    "http://yann.lecun.com/exdb/mnist"
)

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

# expected sizes in bytes (approximate minimums)
MIN_SIZES=(9000000 50000 1500000 4000)

# ── download function ──────────────────────────────────────────────
download_file() {
    local FILE=$1
    local MIN_SIZE=$2
    local EXTRACTED="${FILE%.gz}"

    # skip if already extracted and big enough
    if [ -f "$EXTRACTED" ]; then
        SIZE=$(wc -c < "$EXTRACTED")
        if [ "$SIZE" -gt "$MIN_SIZE" ]; then
            echo "  ✓ Already exists: $EXTRACTED ($(du -h "$EXTRACTED" | cut -f1))"
            return 0
        fi
    fi

    # try each mirror
    for MIRROR in "${MIRRORS[@]}"; do
        echo "  → Trying: $MIRROR/$FILE"
        rm -f "$FILE"

        # try wget first, then curl
        if command -v wget &>/dev/null; then
            wget -q --timeout=30 --tries=2 \
                 --show-progress \
                 "$MIRROR/$FILE" -O "$FILE" 2>&1 || true
        elif command -v curl &>/dev/null; then
            curl -L --silent --show-error \
                 --connect-timeout 30 \
                 "$MIRROR/$FILE" -o "$FILE" 2>&1 || true
        else
            echo "  ERROR: Neither wget nor curl found!"
            exit 1
        fi

        # check file size
        if [ -f "$FILE" ]; then
            SIZE=$(wc -c < "$FILE")
            if [ "$SIZE" -gt 1000 ]; then
                echo "  ✓ Downloaded: $FILE ($(du -h "$FILE" | cut -f1))"
                echo "  → Extracting..."
                gunzip -f "$FILE"
                echo "  ✓ Extracted: $EXTRACTED"
                return 0
            else
                echo "  ✗ Failed (file too small: $SIZE bytes)"
                rm -f "$FILE"
            fi
        fi
    done

    echo ""
    echo "  ERROR: Could not download $FILE from any mirror!"
    echo "  Please download manually from:"
    echo "    https://ossci-datasets.s3.amazonaws.com/mnist/$FILE"
    echo "  Then run: gunzip mnist/$FILE"
    return 1
}

# ── download all files ─────────────────────────────────────────────
echo ""
echo "Downloading 4 MNIST files..."
echo ""

FAILED=0
for i in "${!FILES[@]}"; do
    download_file "${FILES[$i]}" "${MIN_SIZES[$i]}" || FAILED=1
    echo ""
done

# ── verify ────────────────────────────────────────────────────────
echo "Verifying files..."
echo ""
ALL_OK=1
EXPECTED=(
    "train-images-idx3-ubyte:47MB:60000 train images"
    "train-labels-idx1-ubyte:58KB:60000 train labels"
    "t10k-images-idx3-ubyte:8MB:10000 test images"
    "t10k-labels-idx1-ubyte:10KB:10000 test labels"
)

for ENTRY in "${EXPECTED[@]}"; do
    NAME="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    SIZE="${REST%%:*}"
    DESC="${REST#*:}"
    if [ -f "$NAME" ]; then
        echo "  ✓ $NAME (~$SIZE) — $DESC"
    else
        echo "  ✗ MISSING: $NAME"
        ALL_OK=0
    fi
done

echo ""
if [ "$ALL_OK" -eq 1 ] && [ "$FAILED" -eq 0 ]; then
    echo "✓ All MNIST files ready!"
    echo ""
    echo "Now run:"
    echo "  cd .."
    echo "  make mnist_demo"
    echo "  ./mnist_demo"
else
    echo "✗ Some files are missing. Check errors above."
    exit 1
fi
