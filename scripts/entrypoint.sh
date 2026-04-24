#!/bin/bash
set -euo pipefail

# ---- Fetch non-default language bundle on first start ----------------------
# Latin is the only language shipped in the image (flat-layout
# /app/models/{det,rec,cls}.onnx + /app/models/keys.txt). Every other
# language listed below is pulled on demand from the pinned PP-OCRv5 ONNX
# mirror into /app/models/rec/<lang>/. The Dockerfile symlinks
# /app/models/rec → /home/ocr/.cache/turbo-ocr/models/rec so bundles land
# in the same volume that caches TRT engines, and survive container
# restarts.
#
# Setting OCR_SERVER=1 with OCR_LANG=chinese downloads the 84 MB server rec
# instead of the default 16 MB mobile rec. Ignored for other languages.
SUPPORTED_LANGS="arabic chinese eslav greek korean latin thai"

if [[ -n "${OCR_LANG:-}" && "${OCR_LANG}" != "latin" ]]; then
  # Guard against typos before touching the network.
  if ! grep -qw "${OCR_LANG}" <<<"${SUPPORTED_LANGS}"; then
    echo "[entrypoint] FATAL: OCR_LANG='${OCR_LANG}' is not a supported language." >&2
    echo "[entrypoint]        Supported: ${SUPPORTED_LANGS}" >&2
    exit 1
  fi

  REC_ONNX="/app/models/rec/${OCR_LANG}/rec.onnx"
  if [[ ! -f "${REC_ONNX}" ]]; then
    echo "[entrypoint] OCR_LANG=${OCR_LANG} requested, fetching bundle…"
    bash /app/scripts/download_models.sh --lang "${OCR_LANG}" ${OCR_SERVER:+--server}
    # chown only when we own uid 0 — the Dockerfile installs the ocr user
    # and today the base image runs as root, but this stays correct if that
    # ever changes.
    if [[ $EUID -eq 0 ]]; then
      chown -R ocr:ocr /app/models
    fi
  else
    echo "[entrypoint] OCR_LANG=${OCR_LANG} bundle already present, skipping download"
  fi
fi

# Start nginx reverse proxy (absorbs connection storms, keep-alive to Drogon)
nginx -c /app/docker/nginx.conf

# Drop to non-root user and run the OCR server
# TRT engines are auto-built from ONNX on first startup (cached by TRT version + model hash)
exec gosu ocr "$@"
