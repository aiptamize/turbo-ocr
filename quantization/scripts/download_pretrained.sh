#!/bin/bash
# =============================================================================
# Download PP-OCRv5 latin mobile rec TRAINING weights
# =============================================================================
#
# QAT (Quantization-Aware Training) requires the pretrained TRAINING weights
# (.pdparams), NOT the inference model (.pdiparams/.json from HuggingFace).
#
# The HuggingFace repos (PaddlePaddle/latin_PP-OCRv5_mobile_rec) only contain
# inference model files. The training weights are hosted on Baidu's CDN as
# single .pdparams files used by PaddleOCR's training scripts.
#
# Usage:
#   cd /home/nataell/code/epAiland/paddle-highspeed-cpp/quantization
#   bash scripts/download_pretrained.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUANT_DIR="$(dirname "$SCRIPT_DIR")"
PRETRAINED_DIR="${QUANT_DIR}/pretrained/latin_PP-OCRv5_mobile_rec"

mkdir -p "$PRETRAINED_DIR"

# The official PaddleOCR pretrained weights URL from Baidu CDN.
# This is a single .pdparams file containing the trained model parameters.
# Source: PaddleOCR text recognition module documentation
#   https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html
PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams"

# PaddleOCR's load_model() expects a path WITHOUT the .pdparams extension.
# It internally appends .pdparams when loading. So we save as best_accuracy.pdparams
# and reference the path as .../best_accuracy in the config.
OUTPUT_FILE="${PRETRAINED_DIR}/best_accuracy.pdparams"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Pretrained weights already downloaded at: $OUTPUT_FILE"
    echo "  Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    exit 0
fi

echo "============================================"
echo "Downloading PP-OCRv5 Latin Rec Training Weights"
echo "============================================"
echo ""
echo "URL: $PRETRAINED_URL"
echo "Destination: $OUTPUT_FILE"
echo ""
echo "NOTE: This downloads the TRAINING weights (.pdparams), which are"
echo "      required for QAT. The HuggingFace inference models won't work."
echo ""

# Download with wget (resume support)
if command -v wget &>/dev/null; then
    wget -c "$PRETRAINED_URL" -O "$OUTPUT_FILE"
elif command -v curl &>/dev/null; then
    curl -L -C - "$PRETRAINED_URL" -o "$OUTPUT_FILE"
else
    echo "ERROR: Neither wget nor curl found. Install one of them."
    exit 1
fi

# Verify download
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo ""
        echo "WARNING: Downloaded file is suspiciously small ($(du -h "$OUTPUT_FILE" | cut -f1))."
        echo "The download may have failed. Try again or check the URL manually."
        rm -f "$OUTPUT_FILE"
        exit 1
    fi
    echo ""
    echo "Success! Training weights downloaded."
    echo "  File: $OUTPUT_FILE"
    echo "  Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "PaddleOCR load_model() path (without .pdparams extension):"
    echo "  ${OUTPUT_FILE%.pdparams}"
else
    echo "ERROR: Download failed."
    exit 1
fi
