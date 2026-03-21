#!/bin/bash
# =============================================================================
# Quantization-Aware Training (QAT) for PP-OCRv5 Latin Recognition
# =============================================================================
#
# This script performs QAT on the PP-OCRv5 latin mobile recognition model
# to produce an INT8-quantized model for TensorRT deployment.
#
# What is QAT?
#   QAT inserts fake-quantization nodes (simulating INT8 rounding) into the
#   FP32 model, then fine-tunes for several epochs so the weights adapt to
#   quantization noise. This produces significantly better INT8 accuracy than
#   post-training quantization (PTQ). PaddleSlim implements QAT with PACT
#   (Parameterized Clipping Activation Training) which learns optimal
#   clipping ranges during training.
#
# Prerequisites:
#   - GPU with >= 16GB VRAM (PACT roughly doubles memory vs normal training)
#   - PaddlePaddle GPU >= 2.6 (paddlepaddle-gpu)
#   - PaddleSlim >= 2.6
#   - paddle2onnx >= 1.2
#   - Dataset prepared (run scripts/download_dataset.py first)
#   - Pretrained weights downloaded (run scripts/download_pretrained.sh first)
#
# Expected training time:
#   - 10 epochs on 27,000 images: 1-4 hours depending on GPU
#   - RTX 3090/4090: ~1-2 hours
#   - V100: ~2-4 hours
#
# Expected GPU memory:
#   - ~12-16 GB (batch_size=64 with PACT overhead)
#   - If OOM, reduce batch_size_per_card in configs/qat_latin_rec.yml
#
# Usage:
#   cd /home/nataell/code/epAiland/paddle-highspeed-cpp/quantization
#   bash scripts/run_qat.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUANT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$QUANT_DIR")"

cd "$QUANT_DIR"

# ---- Activate Python 3.10 venv (PaddlePaddle requires Python <= 3.12) ----
VENV_DIR="${QUANT_DIR}/.venv-qat-310"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: ${VENV_DIR} ($(python3 --version))"
fi

# ---- Configuration ----

# PaddleOCR source (will be cloned if not found).
# Main branch contains PP-OCRv5 support and deploy/slim/quantization scripts.
PADDLEOCR_DIR="${QUANT_DIR}/PaddleOCR"

# Pretrained model: PP-OCRv5 latin mobile rec training weights.
# Downloaded by scripts/download_pretrained.sh from Baidu CDN.
# PaddleOCR's load_model() expects path WITHOUT .pdparams extension.
PRETRAINED_MODEL="${QUANT_DIR}/pretrained/latin_PP-OCRv5_mobile_rec/best_accuracy"

# Character dictionary (836 latin characters).
DICT_PATH="${PROJECT_DIR}/models/v5_latin_rec/keys.txt"

# QAT config (architecture must match pretrained model exactly).
CONFIG="${QUANT_DIR}/configs/qat_latin_rec.yml"

# Output directories.
OUTPUT_DIR="${QUANT_DIR}/output/qat_latin_rec"
INFERENCE_DIR="${QUANT_DIR}/output/qat_rec_inference"

# ==========================================================================
# Step 0: Validate environment
# ==========================================================================

echo "============================================"
echo "PP-OCRv5 Latin Rec - Quantization-Aware Training"
echo "============================================"
echo ""

# Check Python + PaddlePaddle
python3 -c "import paddle; print(f'PaddlePaddle {paddle.__version__} (GPU: {paddle.is_compiled_with_cuda()})')" 2>/dev/null || {
    echo "ERROR: PaddlePaddle not found or not importable."
    echo ""
    echo "Install PaddlePaddle GPU:"
    echo "  pip install paddlepaddle-gpu"
    echo ""
    echo "See: https://www.paddlepaddle.org.cn/install/quick"
    exit 1
}

# Check PaddleSlim
python3 -c "import paddleslim; print('PaddleSlim OK')" 2>/dev/null || {
    echo "ERROR: PaddleSlim not found."
    echo ""
    echo "Install PaddleSlim:"
    echo "  pip install paddleslim"
    exit 1
}

# Check paddle2onnx
python3 -c "import paddle2onnx; print(f'paddle2onnx {paddle2onnx.__version__}')" 2>/dev/null || {
    echo "WARNING: paddle2onnx not found. ONNX export step will fail."
    echo "Install: pip install paddle2onnx"
}

# Verify GPU is available
python3 -c "
import paddle
assert paddle.is_compiled_with_cuda(), 'PaddlePaddle is CPU-only. QAT requires GPU.'
print(f'GPU: {paddle.device.cuda.device_count()} device(s)')
" || {
    echo "ERROR: No GPU available. QAT requires a CUDA GPU."
    exit 1
}

# Verify data exists
if [ ! -f "${QUANT_DIR}/data/train_list.txt" ]; then
    echo "ERROR: Training data not found."
    echo "Run first:  python3 scripts/download_dataset.py"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "${QUANT_DIR}/data/train_list.txt")
VAL_COUNT=$(wc -l < "${QUANT_DIR}/data/val_list.txt")
echo "Dataset:    ${TRAIN_COUNT} train + ${VAL_COUNT} val samples"

# Verify pretrained model
if [ ! -f "${PRETRAINED_MODEL}.pdparams" ]; then
    echo "ERROR: Pretrained weights not found at: ${PRETRAINED_MODEL}.pdparams"
    echo ""
    echo "Run first:  bash scripts/download_pretrained.sh"
    exit 1
fi

# Verify dictionary
if [ ! -f "$DICT_PATH" ]; then
    echo "ERROR: Character dictionary not found at: $DICT_PATH"
    exit 1
fi

DICT_SIZE=$(wc -l < "$DICT_PATH")
echo "Dictionary: ${DICT_SIZE} characters"

# ==========================================================================
# Step 1: Clone PaddleOCR if needed
# ==========================================================================

if [ ! -d "$PADDLEOCR_DIR" ]; then
    echo ""
    echo "=== Cloning PaddleOCR (main branch) ==="
    echo ""
    git clone --depth 1 --branch main \
        https://github.com/PaddlePaddle/PaddleOCR.git "$PADDLEOCR_DIR"
fi

# Verify quant.py and export_model.py exist in the clone
for SCRIPT in "deploy/slim/quantization/quant.py" "deploy/slim/quantization/export_model.py"; do
    if [ ! -f "${PADDLEOCR_DIR}/${SCRIPT}" ]; then
        echo "ERROR: ${SCRIPT} not found in PaddleOCR clone."
        echo "The repository structure may have changed."
        exit 1
    fi
done

echo ""
echo "Config:     $CONFIG"
echo "Pretrained: ${PRETRAINED_MODEL}.pdparams"
echo "Dict:       $DICT_PATH (${DICT_SIZE} chars)"
echo "Output:     $OUTPUT_DIR"
echo ""
echo "Architecture: SVTR_LCNet / PPLCNetV3 (scale 0.95) / MultiHead (CTC+NRTR)"
echo "QAT method:   PaddleSlim QAT with PACT activation clipping"
echo "Quantization: INT8 weights (channel_wise_abs_max) + INT8 activations (moving_average_abs_max)"
echo "Layers:       Conv2D, Linear"
echo ""

# ==========================================================================
# Step 2: Run QAT Training
# ==========================================================================
#
# PaddleOCR's quant.py does the following internally:
#   1. Builds the model from config (Architecture section)
#   2. Loads pretrained FP32 weights
#   3. Applies PaddleSlim QAT with PACT activation clipping:
#      - Inserts fake-quantization nodes into Conv2D and Linear layers
#      - PACT learns clipping ranges (alpha initialized to 20, L2 decay 2e-5)
#   4. Fine-tunes for epoch_num epochs while weights adapt to quantization noise
#   5. Saves checkpoints with quantization-aware weights
#
# The quant_config is hardcoded in quant.py (not in our YAML):
#   weight_quantize_type: channel_wise_abs_max
#   activation_quantize_type: moving_average_abs_max
#   weight_bits: 8, activation_bits: 8, dtype: int8
#   quantizable_layer_type: [Conv2D, Linear]

echo "=== Step 2: Quantization-Aware Training (10 epochs) ==="
echo ""
echo "This will take 1-4 hours depending on GPU."
echo "Monitor GPU memory -- if OOM, reduce batch_size_per_card in configs/qat_latin_rec.yml"
echo ""

cd "$PADDLEOCR_DIR"

python3 deploy/slim/quantization/quant.py \
    -c "$CONFIG" \
    -o Global.pretrained_model="$PRETRAINED_MODEL" \
       Global.character_dict_path="$DICT_PATH" \
       Global.save_model_dir="$OUTPUT_DIR"

echo ""
echo "QAT training complete."
echo "Best model at: ${OUTPUT_DIR}/best_accuracy"
echo ""

# ==========================================================================
# Step 3: Export Quantized Inference Model
# ==========================================================================
#
# export_model.py re-applies the same QAT config (hardcoded, matching quant.py),
# loads the QAT-trained checkpoints, runs a validation pass, and exports a
# Paddle inference model with INT8-calibrated weight ranges baked in.
#
# Output files:
#   inference.pdmodel   -- model graph with quantization ops
#   inference.pdiparams -- weights (FP32 storage but with INT8-calibrated ranges)

echo "=== Step 3: Export Quantized Inference Model ==="
echo ""

python3 deploy/slim/quantization/export_model.py \
    -c "$CONFIG" \
    -o Global.checkpoints="${OUTPUT_DIR}/best_accuracy" \
       Global.character_dict_path="$DICT_PATH" \
       Global.save_inference_dir="$INFERENCE_DIR"

echo ""
echo "Quantized inference model exported to: $INFERENCE_DIR"
echo ""

# ==========================================================================
# Step 4: Convert to ONNX
# ==========================================================================
#
# paddle2onnx converts the Paddle inference model to ONNX.
# The quantization information (INT8 ranges) is preserved as
# QuantizeLinear/DequantizeLinear nodes that TensorRT consumes.

echo "=== Step 4: Convert to ONNX ==="
echo ""

cd "$QUANT_DIR"

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
echo ""
echo "Artifacts:"
echo "  Training checkpoints: ${OUTPUT_DIR}/"
echo "  Inference model:      ${INFERENCE_DIR}/"
echo "  ONNX model:           ${ONNX_OUTPUT}"
echo ""
echo "============================================"
echo "Next Steps"
echo "============================================"
echo ""
echo "1. Build TensorRT INT8 engine from the ONNX model:"
echo ""
echo "   trtexec --onnx=${ONNX_OUTPUT} \\"
echo "     --saveEngine=output/qat_rec_int8.trt \\"
echo "     --int8 --fp16 \\"
echo "     --minShapes=x:1x3x48x160 \\"
echo "     --optShapes=x:1x3x48x320 \\"
echo "     --maxShapes=x:8x3x48x3200"
echo ""
echo "   Or use the project's convert script:"
echo "     python3 ${PROJECT_DIR}/scripts/convert_onnx_to_trt.py \\"
echo "       --model ${ONNX_OUTPUT} \\"
echo "       --output output/qat_rec_int8.trt \\"
echo "       --type rec"
echo ""
echo "2. Deploy the INT8 engine in turbo-ocr:"
echo ""
echo "   cp output/qat_rec_int8.trt ${PROJECT_DIR}/models/rec.trt"
echo "   # Restart the turbo-ocr server"
echo ""
echo "3. Validate accuracy hasn't degraded significantly:"
echo ""
echo "   python3 ${PROJECT_DIR}/scripts/validate_accuracy.py \\"
echo "     --endpoint http://localhost:8000/ocr"
echo ""
