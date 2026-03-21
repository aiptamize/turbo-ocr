#!/usr/bin/env python3
"""INT8 quantization of PP-OCRv5 latin rec ONNX model via ONNX Runtime static quantization.

This script bypasses PaddlePaddle entirely (which doesn't support RTX 5090 / SM 120).
Instead it uses onnxruntime.quantization to:
  1. Run calibration with real training images (collecting activation ranges)
  2. Insert QuantizeLinear/DequantizeLinear (Q/DQ) nodes into the ONNX model
  3. Output a quantized ONNX model that TensorRT can consume with --int8

Usage:
    cd /home/nataell/code/epAiland/paddle-highspeed-cpp/quantization
    python3 scripts/quantize_onnx_int8.py

    # Then build TensorRT engine (inside Docker or where TRT is available):
    python3 ../scripts/convert_onnx_to_trt.py \
        --model output/rec_int8_qdq.onnx \
        --output ../models/v5_latin_rec/rec_int8.trt \
        --type rec --int8

Prerequisites:
    pip install onnxruntime onnx Pillow numpy
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
    CalibrationMethod,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
QUANT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_ONNX = str(PROJECT_DIR / "models" / "v5_latin_rec" / "rec.onnx")
DEFAULT_OUTPUT = str(QUANT_DIR / "output" / "rec_int8_qdq.onnx")
DEFAULT_DATA_DIR = str(QUANT_DIR / "data" / "train")
DEFAULT_DATA_LIST = str(QUANT_DIR / "data" / "train_list.txt")
DEFAULT_IMG_H = 48
DEFAULT_IMG_W = 320
DEFAULT_CALIB_SIZE = 500  # Number of calibration images (500 is plenty for PTQ)


# ---------------------------------------------------------------------------
# Preprocessing (must match the C++ pipeline's PaddleRec preprocessing)
# ---------------------------------------------------------------------------
def preprocess_image(img_path: str, img_h: int = 48, img_w: int = 320) -> np.ndarray:
    """Preprocess a single OCR crop image to model input format.

    Matches PaddleOCR's rec preprocessing:
      1. Resize to (img_h, proportional_width), keeping aspect ratio
      2. If proportional_width > img_w, resize to (img_h, img_w)
      3. Pad to img_w with zeros on the right
      4. Normalize to [-1, 1] range: (pixel/255 - 0.5) / 0.5
      5. CHW format, float32
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Compute proportional width
    ratio = img_w / max(w, 1)
    ratio_h = img_h / max(h, 1)
    if ratio_h < ratio:
        ratio = ratio_h

    resize_w = int(w * ratio)
    resize_w = max(1, min(resize_w, img_w))

    img = img.resize((resize_w, img_h), Image.BILINEAR)

    # Create padded image
    padded = np.zeros((img_h, img_w, 3), dtype=np.float32)
    img_np = np.array(img, dtype=np.float32)
    padded[:, :resize_w, :] = img_np[:, :resize_w, :]

    # Normalize to [-1, 1]
    padded = (padded / 255.0 - 0.5) / 0.5

    # HWC -> CHW
    padded = padded.transpose(2, 0, 1)

    return padded


# ---------------------------------------------------------------------------
# Calibration data reader for onnxruntime quantization
# ---------------------------------------------------------------------------
class RecCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed OCR images to the quantization calibrator."""

    def __init__(
        self,
        data_dir: str,
        data_list: str,
        input_name: str = "x",
        img_h: int = DEFAULT_IMG_H,
        img_w: int = DEFAULT_IMG_W,
        max_samples: int = DEFAULT_CALIB_SIZE,
    ):
        self.data_dir = data_dir
        self.input_name = input_name
        self.img_h = img_h
        self.img_w = img_w

        # Load image paths from the data list
        self.image_paths = []
        with open(data_list, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    self.image_paths.append(parts[0])

        # Limit calibration set size
        if len(self.image_paths) > max_samples:
            # Take evenly spaced samples for better coverage
            step = len(self.image_paths) / max_samples
            indices = [int(i * step) for i in range(max_samples)]
            self.image_paths = [self.image_paths[i] for i in indices]

        print(f"Calibration dataset: {len(self.image_paths)} images from {data_dir}")
        self.index = 0
        self.total = len(self.image_paths)

    def get_next(self):
        if self.index >= self.total:
            return None

        img_path = os.path.join(self.data_dir, self.image_paths[self.index])
        self.index += 1

        if self.index % 100 == 0 or self.index == self.total:
            print(f"  Calibration progress: {self.index}/{self.total}")

        try:
            img = preprocess_image(img_path, self.img_h, self.img_w)
            # Batch dimension
            img = np.expand_dims(img, axis=0).astype(np.float32)
            return {self.input_name: img}
        except Exception as e:
            print(f"  Warning: Failed to load {img_path}: {e}")
            return self.get_next()  # Skip bad images

    def rewind(self):
        self.index = 0


# ---------------------------------------------------------------------------
# ONNX model preparation
# ---------------------------------------------------------------------------
def prepare_model_for_quantization(onnx_path: str, prepared_path: str) -> str:
    """Prepare ONNX model for quantization (shape inference + optimization).

    onnxruntime.quantization requires models to have complete shape information.
    This step runs shape inference and basic optimizations.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print(f"Preparing model for quantization: {onnx_path}")
    print(f"  -> {prepared_path}")

    quant_pre_process(
        input_model_path=onnx_path,
        output_model_path=prepared_path,
        auto_merge=True,
    )

    # Verify the prepared model loads
    model = onnx.load(prepared_path)
    onnx.checker.check_model(model, full_check=True)
    print(f"  Prepared model validated OK ({len(model.graph.node)} nodes)")

    return prepared_path


# ---------------------------------------------------------------------------
# Main quantization pipeline
# ---------------------------------------------------------------------------
def quantize_model(
    onnx_path: str,
    output_path: str,
    data_dir: str,
    data_list: str,
    img_h: int = DEFAULT_IMG_H,
    img_w: int = DEFAULT_IMG_W,
    calib_size: int = DEFAULT_CALIB_SIZE,
    calibration_method: str = "entropy",
    per_channel: bool = True,
):
    """Run static INT8 quantization on the ONNX recognition model.

    Args:
        onnx_path: Path to input FP32 ONNX model
        output_path: Path for output INT8 Q/DQ ONNX model
        data_dir: Directory containing calibration images
        data_list: TSV file listing image filenames and labels
        img_h: Input image height (default 48)
        img_w: Input image width (default 320)
        calib_size: Number of calibration images to use
        calibration_method: 'entropy' (best accuracy) or 'minmax' (fastest)
        per_channel: Use per-channel weight quantization (better accuracy)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Step 1: Determine input name from the ONNX model
    print("\n=== Step 1: Inspect ONNX model ===")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print(f"  Input:  {input_name} {input_shape}")
    print(f"  Output: {output_name} {output_shape}")
    del session

    # Step 2: Prepare model (shape inference)
    print("\n=== Step 2: Prepare model for quantization ===")
    prepared_path = output_path.replace(".onnx", "_prepared.onnx")
    prepare_model_for_quantization(onnx_path, prepared_path)

    # Step 3: Create calibration data reader
    print("\n=== Step 3: Calibration ===")
    calib_reader = RecCalibrationDataReader(
        data_dir=data_dir,
        data_list=data_list,
        input_name=input_name,
        img_h=img_h,
        img_w=img_w,
        max_samples=calib_size,
    )

    # Step 4: Select calibration method
    if calibration_method == "entropy":
        calib_method = CalibrationMethod.Entropy
        print("  Using Entropy calibration (best accuracy, slower)")
    elif calibration_method == "percentile":
        calib_method = CalibrationMethod.Percentile
        print("  Using Percentile calibration")
    else:
        calib_method = CalibrationMethod.MinMax
        print("  Using MinMax calibration (fastest, slightly lower accuracy)")

    # Step 5: Run quantization
    print(f"\n=== Step 4: Quantize (per_channel={per_channel}) ===")
    t0 = time.time()

    # Nodes to exclude from quantization (softmax outputs, final classification)
    # These are better kept in FP16/FP32 for accuracy
    nodes_to_exclude = []

    # Load the prepared model to find softmax nodes
    prep_model = onnx.load(prepared_path)
    for node in prep_model.graph.node:
        if node.op_type in ("Softmax", "LayerNormalization"):
            nodes_to_exclude.append(node.name)
    del prep_model

    if nodes_to_exclude:
        print(f"  Excluding {len(nodes_to_exclude)} sensitive nodes from quantization:")
        for n in nodes_to_exclude[:5]:
            print(f"    - {n}")
        if len(nodes_to_exclude) > 5:
            print(f"    ... and {len(nodes_to_exclude) - 5} more")

    extra_opts = {
        "ActivationSymmetric": True,  # Symmetric quantization for TRT compatibility
        "WeightSymmetric": True,
    }
    if calibration_method == "entropy":
        extra_opts["CalibMovingAverage"] = True

    quantize_static(
        model_input=prepared_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,  # Q/DQ nodes that TensorRT understands
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=calib_method,
        nodes_to_exclude=nodes_to_exclude,
        extra_options=extra_opts,
    )

    elapsed = time.time() - t0
    print(f"\n  Quantization completed in {elapsed:.1f}s")

    # Step 6: Verify output model
    print("\n=== Step 5: Verify quantized model ===")
    quant_model = onnx.load(output_path)
    onnx.checker.check_model(quant_model, full_check=True)

    # Count Q/DQ nodes
    qdq_count = sum(
        1 for n in quant_model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    total_nodes = len(quant_model.graph.node)
    print(f"  Quantized model: {total_nodes} nodes ({qdq_count} Q/DQ nodes)")
    print(f"  Output: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Step 7: Quick accuracy check -- run a sample through both models
    print("\n=== Step 6: Quick accuracy comparison ===")
    verify_quantized_model(onnx_path, output_path, data_dir, data_list, input_name, img_h, img_w)

    # Cleanup prepared model
    if os.path.exists(prepared_path):
        os.remove(prepared_path)
        print(f"\n  Cleaned up: {prepared_path}")

    print(f"\n{'='*60}")
    print("Quantization complete!")
    print(f"{'='*60}")
    print(f"\nQuantized ONNX model: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Build TensorRT INT8 engine:")
    print(f"     python3 {PROJECT_DIR}/scripts/convert_onnx_to_trt.py \\")
    print(f"       --model {output_path} \\")
    print(f"       --output {PROJECT_DIR}/models/v5_latin_rec/rec_int8.trt \\")
    print(f"       --type rec --int8")
    print(f"")
    print(f"  2. Deploy (replace rec.trt with rec_int8.trt):")
    print(f"     cp {PROJECT_DIR}/models/v5_latin_rec/rec_int8.trt {PROJECT_DIR}/models/rec.trt")
    print(f"     # Restart the turbo-ocr server")
    print(f"")
    print(f"  3. Validate accuracy:")
    print(f"     python3 {PROJECT_DIR}/scripts/validate_accuracy.py \\")
    print(f"       --endpoint http://localhost:8000/ocr")


def verify_quantized_model(
    fp32_path: str,
    int8_path: str,
    data_dir: str,
    data_list: str,
    input_name: str,
    img_h: int,
    img_w: int,
    num_samples: int = 20,
):
    """Compare FP32 and INT8 model outputs on a few samples."""
    # Load label list
    samples = []
    with open(data_list) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                samples.append((parts[0], parts[1]))
    samples = samples[:num_samples]

    # Load both models
    fp32_session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    fp32_correct = 0
    int8_correct = 0
    max_diff_sum = 0.0

    # Load character dictionary
    keys_path = str(PROJECT_DIR / "models" / "v5_latin_rec" / "keys.txt")
    chars = ["blank"]  # CTC blank at index 0
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            for line in f:
                chars.append(line.strip())
    chars.append(" ")  # space at end

    for img_file, gt_label in samples:
        img_path = os.path.join(data_dir, img_file)
        if not os.path.exists(img_path):
            continue

        img = preprocess_image(img_path, img_h, img_w)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        fp32_out = fp32_session.run(None, {input_name: img})[0]  # [1, seq, classes]
        int8_out = int8_session.run(None, {input_name: img})[0]

        # CTC greedy decode
        def ctc_decode(logits):
            # logits shape: [1, seq_len, num_classes]
            indices = np.argmax(logits[0], axis=-1)
            # Remove blanks and duplicates
            result = []
            prev = -1
            for idx in indices:
                if idx != 0 and idx != prev:  # 0 = blank
                    if idx < len(chars):
                        result.append(chars[idx])
                    else:
                        result.append("?")
                prev = idx
            return "".join(result)

        fp32_text = ctc_decode(fp32_out)
        int8_text = ctc_decode(int8_out)

        if fp32_text == gt_label:
            fp32_correct += 1
        if int8_text == gt_label:
            int8_correct += 1

        # Max absolute difference in logits
        max_diff = np.max(np.abs(fp32_out - int8_out))
        max_diff_sum += max_diff

    n = len(samples)
    print(f"  Samples:     {n}")
    print(f"  FP32 correct: {fp32_correct}/{n} ({100*fp32_correct/max(n,1):.0f}%)")
    print(f"  INT8 correct: {int8_correct}/{n} ({100*int8_correct/max(n,1):.0f}%)")
    print(f"  Avg max logit diff: {max_diff_sum/max(n,1):.4f}")

    if int8_correct < fp32_correct * 0.8:
        print("  WARNING: INT8 accuracy dropped significantly!")
        print("  Consider using more calibration samples or entropy calibration.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="INT8 quantize PP-OCRv5 latin rec ONNX model using onnxruntime"
    )
    parser.add_argument(
        "--model", default=DEFAULT_ONNX,
        help=f"Input FP32 ONNX model (default: {DEFAULT_ONNX})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output INT8 Q/DQ ONNX model (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Calibration image directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--data-list", default=DEFAULT_DATA_LIST,
        help=f"Image list file (default: {DEFAULT_DATA_LIST})"
    )
    parser.add_argument(
        "--calib-size", type=int, default=DEFAULT_CALIB_SIZE,
        help=f"Number of calibration images (default: {DEFAULT_CALIB_SIZE})"
    )
    parser.add_argument(
        "--calibration-method", choices=["entropy", "minmax", "percentile"],
        default="entropy",
        help="Calibration method (default: entropy)"
    )
    parser.add_argument(
        "--no-per-channel", dest="per_channel", action="store_false", default=True,
        help="Disable per-channel weight quantization"
    )
    parser.add_argument(
        "--img-h", type=int, default=DEFAULT_IMG_H,
        help=f"Input image height (default: {DEFAULT_IMG_H})"
    )
    parser.add_argument(
        "--img-w", type=int, default=DEFAULT_IMG_W,
        help=f"Input image width (default: {DEFAULT_IMG_W})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PP-OCRv5 Latin Rec - INT8 Quantization (ONNX Runtime)")
    print("=" * 60)
    print(f"  Input model:          {args.model}")
    print(f"  Output model:         {args.output}")
    print(f"  Calibration images:   {args.data_dir}")
    print(f"  Calibration size:     {args.calib_size}")
    print(f"  Calibration method:   {args.calibration_method}")
    print(f"  Per-channel weights:  {args.per_channel}")
    print(f"  Input size:           {args.img_h}x{args.img_w}")
    print()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)
    if not os.path.exists(args.data_list):
        print(f"ERROR: Data list not found: {args.data_list}")
        sys.exit(1)

    quantize_model(
        onnx_path=args.model,
        output_path=args.output,
        data_dir=args.data_dir,
        data_list=args.data_list,
        img_h=args.img_h,
        img_w=args.img_w,
        calib_size=args.calib_size,
        calibration_method=args.calibration_method,
        per_channel=args.per_channel,
    )


if __name__ == "__main__":
    main()
