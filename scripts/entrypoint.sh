#!/bin/bash
set -euo pipefail

# Build TRT engines on first run (needs GPU)
if [ ! -f /app/models/det.trt ] || [ /app/models/det.onnx -nt /app/models/det.trt ]; then
  echo "Building TRT engines (first run)..."
  python3 /app/scripts/convert_onnx_to_trt.py --model /app/models/det.onnx --output /app/models/det.trt --type det
  python3 /app/scripts/convert_onnx_to_trt.py --model /app/models/rec.onnx --output /app/models/rec.trt --type rec
  python3 /app/scripts/convert_onnx_to_trt.py --model /app/models/cls.onnx --output /app/models/cls.trt --type cls
  echo "TRT engines built."
fi

exec "$@"
