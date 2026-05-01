#!/bin/bash
set -euo pipefail

# ---- Validate OCR_LANG and ensure the bundle is on disk --------------------
# The Dockerfile bakes every supported bundle at build time via
# fetch_release_models.sh (flat /app/models/{det,rec,cls}.onnx + keys.txt
# for Latin; /app/models/rec/<lang>/ for every other script). So in a
# normal deployment the `[[ ! -f ]]` branch below never fires — it exists
# as a self-heal path in case /app/models gets mounted over with an empty
# volume, or the bundle is deleted.
#
# Setting OCR_SERVER=1 with OCR_LANG=chinese selects the 84 MB server rec
# variant instead of the default 16 MB mobile rec. Ignored for other
# languages.
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

# Render nginx config from template — substitutes ${MAX_BODY_MB} so the
# proxy and the C++ servers (which both read MAX_BODY_MB at startup) agree
# on the body cap. Default 100 to match historical behaviour.
export MAX_BODY_MB="${MAX_BODY_MB:-100}"
# Validate up front: matches the C++ env_int(..., 1, 102400) range so the
# nginx config rendered here and the Drogon/gRPC limits inside the
# server agree on the same accepted values.
#  - reject leading zeros / "0"     (nginx interprets `0m` as unlimited)
#  - reject empty / non-numeric     (nginx fails 90s into startup with a confusing parse error)
#  - reject anything > 102400 MB    (matches env_int upper bound)
if ! [[ "$MAX_BODY_MB" =~ ^[1-9][0-9]*$ ]] || (( MAX_BODY_MB > 102400 )); then
  echo "[entrypoint] FATAL: MAX_BODY_MB must be a positive integer in [1, 102400] (got: '$MAX_BODY_MB')" >&2
  exit 1
fi

# ---- Preflight: TRT engine cache must be writable -------------------------
# Mirrors get_engine_cache_dir() in src/engine/onnx_to_trt.cpp:
#   $TRT_ENGINE_CACHE → $HOME/.cache/turbo-ocr → /tmp/turbo-ocr-engines
# A read-only volume mount here makes the first request crash with no clear
# signal; fail fast at startup with an actionable message instead.
if [[ -n "${TRT_ENGINE_CACHE:-}" ]]; then
  TRT_CACHE_DIR="${TRT_ENGINE_CACHE}"
elif [[ -n "${HOME:-}" ]]; then
  TRT_CACHE_DIR="${HOME}/.cache/turbo-ocr"
else
  TRT_CACHE_DIR="/tmp/turbo-ocr-engines"
fi
mkdir -p "${TRT_CACHE_DIR}" 2>/dev/null || true
TRT_CACHE_SENTINEL="${TRT_CACHE_DIR}/.entrypoint_writecheck.$$"
if ! ( : > "${TRT_CACHE_SENTINEL}" ) 2>/dev/null; then
  echo "[entrypoint] FATAL: TRT engine cache directory '${TRT_CACHE_DIR}' is not writable." >&2
  echo "[entrypoint]        Mount it read-write or unset TRT_ENGINE_CACHE to fall back to ~/.cache/turbo-ocr." >&2
  exit 1
fi
rm -f "${TRT_CACHE_SENTINEL}"

NGINX_CONF=/tmp/nginx.conf
envsubst '${MAX_BODY_MB}' < /app/docker/nginx.conf.template > "$NGINX_CONF"

# Start nginx reverse proxy (absorbs connection storms, keep-alive to Drogon)
nginx -c "$NGINX_CONF"

# Drop to non-root user and run the OCR server
# TRT engines are auto-built from ONNX on first startup (cached by TRT version + model hash)
exec gosu ocr "$@"
