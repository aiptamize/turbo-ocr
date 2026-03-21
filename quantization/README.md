# PP-OCRv5 Latin Recognition -- INT8 Quantization-Aware Training (QAT)

This directory contains everything needed to perform **Quantization-Aware Training (QAT)** on the PP-OCRv5 latin mobile recognition model, producing an INT8 model for TensorRT deployment in the turbo-ocr server.

## What is QAT?

**QAT is NOT simple post-training quantization.** It is a GPU training process:

1. The pretrained FP32 model is loaded
2. PaddleSlim inserts **fake-quantization nodes** into every Conv2D and Linear layer, simulating INT8 rounding during forward passes
3. **PACT (Parameterized Clipping Activation Training)** learns optimal clipping ranges for activations -- each layer gets a trainable alpha parameter
4. The model is **fine-tuned for 10 epochs** so weights adapt to quantization noise through backpropagation
5. The result is an INT8 model with calibrated weight ranges that TensorRT can deploy directly

**Why QAT instead of PTQ (post-training quantization)?**
- PTQ just calibrates ranges on a small dataset without any gradient updates
- QAT lets the model compensate for quantization error during training
- QAT typically achieves **<1% accuracy drop** vs FP32, while PTQ can lose 3-10%

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 12 GB | 16 GB |
| Training time | 1 hour | 2-4 hours |
| Disk space | 2 GB (dataset + models) | 4 GB (with checkpoints) |
| PaddlePaddle | paddlepaddle-gpu >= 2.6 | latest |
| PaddleSlim | >= 2.6 | latest |
| paddle2onnx | >= 1.2 | latest |

**PACT roughly doubles GPU memory usage** compared to normal training because it maintains additional parameters and gradients for the learned clipping ranges.

## Model Architecture

The PP-OCRv5 latin mobile rec model:

| Component | Value |
|-----------|-------|
| Algorithm | SVTR_LCNet |
| Backbone | PPLCNetV3 (scale 0.95) |
| Head | MultiHead (CTCHead + NRTRHead) |
| Input shape | 3 x 48 x 320 |
| Character set | 836 latin characters (letters, digits, accented chars, symbols) |
| Loss | MultiLoss (CTCLoss + NRTRLoss) |

## Quick Start

### 1. Prepare the dataset

Downloads 30,000 synthetic word images from MJSynth (~1GB):

```bash
cd quantization/
pip install -r requirements.txt
python3 scripts/download_dataset.py
```

This creates 27,000 training + 3,000 validation images in PaddleOCR format.

### 2. Download pretrained training weights

Downloads the official PP-OCRv5 latin mobile rec `.pdparams` from Baidu CDN (~14MB):

```bash
bash scripts/download_pretrained.sh
```

**Important:** This downloads TRAINING weights (`.pdparams`), not the HuggingFace inference model. QAT requires training weights because it needs to fine-tune through backpropagation.

### 3. Run QAT (requires CUDA GPU)

```bash
# Install GPU dependencies first:
pip install paddlepaddle-gpu paddleslim paddle2onnx

# Run the full QAT pipeline:
bash scripts/run_qat.sh
```

The script will:
1. Verify GPU, PaddlePaddle, and PaddleSlim are available
2. Clone PaddleOCR source (for `deploy/slim/quantization/quant.py`)
3. Run 10 epochs of QAT fine-tuning with PACT activation clipping
4. Export the quantized Paddle inference model
5. Convert to ONNX with quantization ops preserved

### 4. Build TensorRT INT8 engine

```bash
trtexec --onnx=output/qat_rec_int8.onnx \
  --saveEngine=output/qat_rec_int8.trt \
  --int8 --fp16 \
  --minShapes=x:1x3x48x160 \
  --optShapes=x:1x3x48x320 \
  --maxShapes=x:8x3x48x3200
```

Or use the project's convert script:

```bash
python3 ../scripts/convert_onnx_to_trt.py \
  --model output/qat_rec_int8.onnx \
  --output output/qat_rec_int8.trt \
  --type rec
```

### 5. Deploy in turbo-ocr

```bash
cp output/qat_rec_int8.trt ../models/rec.trt
# Restart the turbo-ocr server
```

## How QAT Works (Technical Details)

PaddleOCR's `deploy/slim/quantization/quant.py` uses PaddleSlim's QAT framework:

**Quantization config** (hardcoded in quant.py, not in the YAML config):
```python
quant_config = {
    "weight_quantize_type": "channel_wise_abs_max",   # per-channel for weights
    "activation_quantize_type": "moving_average_abs_max",  # running stats for activations
    "weight_bits": 8,
    "activation_bits": 8,
    "dtype": "int8",
    "window_size": 10000,
    "moving_rate": 0.9,
    "quantizable_layer_type": ["Conv2D", "Linear"],
}
```

**PACT activation clipping:**
- Each quantized layer gets a learnable `alpha` parameter (initialized to 20)
- Activations are clipped to `[-alpha, +alpha]` before quantization
- Alpha is trained with L2 regularization (2e-5) at learning rate 1.0
- This learns tight, layer-specific activation ranges for better INT8 utilization

**QAT training config** (in `configs/qat_latin_rec.yml`):
- Learning rate: 0.0001 (5x lower than original training at 0.0005)
- Epochs: 10 (vs 75 for original training)
- Batch size: 64 per GPU (vs 128 for original, reduced for PACT memory overhead)
- Optimizer: Adam with cosine LR schedule and 1-epoch warmup

## Expected Results

| Metric | FP32 Baseline | INT8 QAT |
|--------|--------------|----------|
| Accuracy (MJSynth val) | ~75-80% | ~74-79% (<1-2% drop) |
| Model size | ~14 MB | ~4 MB (~4x reduction) |
| TensorRT inference | baseline | ~1.5-2x speedup |

Note: MJSynth accuracy appears modest because it contains many challenging synthetic images. On real-world latin text, the model achieves 84.7% (per PaddleOCR benchmarks).

## File Structure

```
quantization/
  configs/
    qat_latin_rec.yml        -- QAT training config (matches PP-OCRv5_mobile_rec architecture)
  scripts/
    download_dataset.py      -- Download MJSynth subset from HuggingFace (streaming)
    download_pretrained.sh   -- Download training weights from Baidu CDN
    run_qat.sh               -- Full QAT pipeline: validate -> train -> export -> ONNX
  data/
    train/                   -- 27,000 training images (created by download_dataset.py)
    val/                     -- 3,000 validation images
    train_list.txt           -- Training labels (filename\tlabel, PaddleOCR SimpleDataSet format)
    val_list.txt             -- Validation labels
  pretrained/                -- Downloaded training weights (created by download_pretrained.sh)
  output/                    -- QAT outputs (checkpoints, inference model, ONNX)
  PaddleOCR/                 -- Cloned PaddleOCR repo (created by run_qat.sh)
  requirements.txt           -- Python dependencies
```

## Troubleshooting

**OOM (Out of Memory):**
Reduce `batch_size_per_card` in `configs/qat_latin_rec.yml`. Try 32 for 12GB GPUs.

**"Pretrained model not found":**
Run `bash scripts/download_pretrained.sh` first. The HuggingFace inference model (`.pdiparams`) will NOT work -- QAT needs training weights (`.pdparams`).

**"PaddleSlim not found":**
Install with `pip install paddleslim`. Make sure it matches your PaddlePaddle version.

**Poor accuracy after QAT:**
- Increase epochs from 10 to 20 in the config
- Lower the learning rate to 0.00005
- Use more training data (increase samples in download_dataset.py)
- Verify the character dictionary matches the pretrained model exactly (836 chars)

**ONNX export fails:**
Ensure `paddle2onnx >= 1.2` is installed. Try opset 14 if 16 causes issues.
