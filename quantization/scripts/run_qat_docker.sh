#!/bin/bash
# =============================================================================
# QAT training script for Docker container
# Paths use /app/ prefix (Docker container layout)
# =============================================================================
set -euo pipefail

QUANT_DIR="/app/quantization"
PADDLEOCR_DIR="/app/PaddleOCR"
PRETRAINED_MODEL="${QUANT_DIR}/pretrained/latin_PP-OCRv5_mobile_rec/best_accuracy"
DICT_PATH="/app/models/v5_latin_rec/keys.txt"
CONFIG="${QUANT_DIR}/configs/qat_latin_rec.yml"
OUTPUT_DIR="${QUANT_DIR}/output/qat_latin_rec"
INFERENCE_DIR="${QUANT_DIR}/output/qat_rec_inference"

echo "============================================"
echo "PP-OCRv5 Latin Rec - QAT (Docker)"
echo "============================================"

python3 -c "
import paddle
print(f'PaddlePaddle {paddle.__version__} (CUDA: {paddle.is_compiled_with_cuda()}, GPUs: {paddle.device.cuda.device_count()})')
import paddleslim
print('PaddleSlim OK')
"

TRAIN_COUNT=$(wc -l < "${QUANT_DIR}/data/train_list.txt")
VAL_COUNT=$(wc -l < "${QUANT_DIR}/data/val_list.txt")
echo "Dataset: ${TRAIN_COUNT} train + ${VAL_COUNT} val"
echo "Dict:    $(wc -l < "$DICT_PATH") characters"
echo ""

# ==========================================================================
# Step 1: QAT Training
# ==========================================================================
echo "=== QAT Training (10 epochs) ==="
cd "$PADDLEOCR_DIR"

python3 deploy/slim/quantization/quant.py \
    -c "$CONFIG" \
    -o Global.pretrained_model="$PRETRAINED_MODEL" \
       Global.character_dict_path="$DICT_PATH" \
       Global.save_model_dir="$OUTPUT_DIR"

echo ""
echo "QAT training complete. Best model: ${OUTPUT_DIR}/best_accuracy"
echo ""

# ==========================================================================
# Step 2: Export Quantized Inference Model
# ==========================================================================
echo "=== Export Quantized Inference Model ==="

python3 deploy/slim/quantization/export_model.py \
    -c "$CONFIG" \
    -o Global.checkpoints="${OUTPUT_DIR}/best_accuracy" \
       Global.character_dict_path="$DICT_PATH" \
       Global.save_inference_dir="$INFERENCE_DIR"

echo "Exported to: $INFERENCE_DIR"
echo ""

# ==========================================================================
# Step 3: Convert to ONNX
# ==========================================================================
echo "=== Convert to ONNX ==="

ONNX_OUTPUT="${QUANT_DIR}/output/qat_rec_int8.onnx"

python3 -m paddle2onnx \
    --model_dir "$INFERENCE_DIR" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file "$ONNX_OUTPUT" \
    --opset_version 16 \
    --enable_onnx_checker true

echo ""
echo "============================================"
echo "QAT Pipeline Complete!"
echo "============================================"
echo "Artifacts in ${QUANT_DIR}/output/:"
ls -lh "${QUANT_DIR}/output/"
echo ""
echo "ONNX model: ${ONNX_OUTPUT}"
if [ -f "$ONNX_OUTPUT" ]; then
    echo "ONNX size: $(du -h "$ONNX_OUTPUT" | cut -f1)"
fi
